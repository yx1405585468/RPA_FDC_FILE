from src.correlation.building_dataframe import BuildSparkDataframe
from src.correlation.by_site_algorithms.compare_inline import CorrCompareInlineBySiteAlgorithm
from src.correlation.by_site_algorithms.compare_wat import CorrCompareWatBySiteAlgorithm
import pyspark
from pyspark.sql import SparkSession
import pandas as pd
from src.correlation.common_process.data_preprocessing import MergeAllDataSourceToResultTable


class CorrCompareBySiteAlgorithm(object):
    algo_dispatch_map = {

        'inline': CorrCompareInlineBySiteAlgorithm,
        'wat': CorrCompareWatBySiteAlgorithm,
    }

    def __init__(self,
                 sparkSession: SparkSession,
                 properties_config: pd.DataFrame,
                 query_sql_dict: dict,
                 base_df: pyspark.sql.DataFrame,
                 config_dict: dict):
        self.sparkSession = sparkSession
        self.properties_config = properties_config
        self.query_sql_dict = query_sql_dict
        self.config_dict = config_dict
        self.base_df = base_df

    def run(self):
        request_id = self.config_dict['request_id']
        result_df_list = []

        print(
            "------------------------------------- by site 算法执行入参配置如下 -------------------------------------")
        # print(f"properties_config: {self.properties_config.to_json()}")
        import json
        print(f"query_sql_dict: {json.dumps(self.query_sql_dict)}")
        print(f"config_dict: {json.dumps(self.config_dict)}")

        for key, value in self.config_dict.items():
            if key == 'request_id':
                continue

            elif key in ['inline', 'wat']:
                if self.config_dict.get(key) is not None:
                    source_df = BuildSparkDataframe.get_dataframe(sparkSession=self.sparkSession,
                                                                  query_sql_dict=self.query_sql_dict,
                                                                  algo_type=key,
                                                                  properties_config=self.properties_config)
                    # 获取数据源对应处理的算法类
                    current_corr_by_site_algorithm = self.algo_dispatch_map[key]

                    if source_df:
                        # 执行bysite关系检测算法
                        result = current_corr_by_site_algorithm.run(
                            base_df=self.base_df,
                            source_df=source_df,
                            config_dict=self.config_dict.get(key),
                            request_id=request_id,
                        )

                        result_df_list.append(result)

        # 各个数据源处理的结果后处理
        return MergeAllDataSourceToResultTable().run(result_df_list)


if __name__ == '__main__':
    import os
    import json
    import warnings
    import pandas as pd
    import pyspark.pandas as ps
    from pyspark.sql import SparkSession
    import numpy as np
    from datetime import datetime
    from pyspark.sql.functions import col


    def get_local_spark():
        import findspark
        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"

        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        # findspark.init()
        # os.environ['PYSPARK_PYTHON'] = "/usr/local/python-3.9.13/bin/python3"
        spark = SparkSession.builder.appName("example").getOrCreate()
        return spark


    def get_remote_spark():
        # os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
        warnings.filterwarnings('ignore')
        #
        os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
        spark = SparkSession.builder \
            .appName("pandas_udf") \
            .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
            .config("spark.scheduler.mode", "FAIR") \
            .config('spark.driver.memory', '8g') \
            .config('spark.driver.cores', '12') \
            .config('spark.executor.memory', '8g') \
            .config('spark.executor.cores', '12') \
            .config('spark.cores.max', '12') \
            .config('spark.driver.host', '192.168.28.49') \
            .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
            .getOrCreate()

        return spark


    spark = get_local_spark()


    # spark = get_remote_spark()

    def test_single_inline(spark):

        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv")

        # 异常表征 df,from uva  => compare inline
        base_df = df_pandas.query("OPE_NO == '1F.FQE10' and PARAMETRIC_NAME == 'FDS1'")
        df_spark = ps.from_pandas(df_pandas[df_pandas['WAFER_ID'].isin(base_df['WAFER_ID'].values.tolist())]).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        select_site_cols = []
        for col_name in base_df.columns:
            if col_name.startswith("SITE") and col_name.endswith("VAL"):
                select_site_cols.append(col_name)

        print('------------------------------')
        print(pd.DataFrame(base_df
                           .filter(items=select_site_cols)

                           .apply(lambda x: x.mean(), axis=0)
                           ).rename(columns={0: 'value'})
              .reset_index(drop=False).rename(columns={'index': 'NAME'})

              )

        base_df_spark = ps.from_pandas(
            pd.DataFrame(base_df
                         .filter(items=select_site_cols)

                         .apply(lambda x: x.mean(), axis=0)
                         ).rename(columns={0: 'value'})
            .reset_index(drop=False).rename(columns={'index': 'NAME'})

        ).to_spark()
        base_site_columns = select_site_cols

        base_df_spark.show()

        #
        # base_df_spark = base_df_spark.agg(
        #     *(
        #         mean(col_name).alias(col_name) for col_name in base_site_columns
        #     )
        # )
        # base_df_spark.show()

        json_loads_dict = {"requestId": "269",

                           "inline": {
                               "group_by_list": ["PRODUCT_ID", "OPE_NO"],
                               "merge_prodg1_list": None,
                               "merge_product_list": None,
                               # "merge_eqp_list": [],
                               # "merge_chamber": [],

                               # "merge_operno_list": [{"xx1": ["1C.CDG10", "1V.EQW10", "1U.PQW10"]},
                               #                 {"xx2": ["1V.PQW10", "1F.FQE10"]}],
                               "merge_operno_list": [],
                               # "mergeOperno": None
                               "base_site_columns": base_site_columns
                           }
                           }

        final_result = CorrCompareInlineBySiteAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                            config_dict=json_loads_dict.get("inline"),
                                                            request_id="269")
        #
        # final_result = final_result.orderBy(col("WEIGHT").desc())
        # # #
        # # # pprint(df_spark.dtypes)
        # final_result.show()
        # #
        return final_result


    def test_single_wat(spark):
        import time

        start = time.time()
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\wat_select.csv")
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(
        #     ["AEMNRM01N.0B01", "AEMNE801N.0B01", "AFXNE001N.0C01", "AGKNCE01N.0A01", "AFXNJ701N.0B01"])]
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(["AEMNRM01N.0B01", "AEMNE801N.0B01"])]

        base_df = df_pandas.query("""
               OPE_NO == 'ST.TTS10' & PARAMETRIC_NAME == 'BVjLN+LN+_I_p14_10nA' & PRODUCT_ID == 'AEMNRM01N.0B01'
               """)

        # 根据wafer id 过滤
        df_spark = ps.from_pandas(df_pandas[df_pandas["WAFER_ID"].isin(base_df["WAFER_ID"].unique())]).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()
        print(base_df)
        select_site_cols = []
        for col_name in base_df.columns:
            if col_name.startswith("SITE") and col_name.endswith("VAL"):
                select_site_cols.append(col_name)

        from pprint import pprint
        pprint(select_site_cols)

        base_df_spark = ps.from_pandas(
            (pd.DataFrame(base_df
                          .filter(items=select_site_cols)

                          .apply(lambda x: x.mean(), axis=0)
                          ).rename(columns={0: 'value'})
             .reset_index(drop=False).rename(columns={'index': 'NAME'})

             )

        ).to_spark()

        base_df_spark.show()

        json_loads_dict = {"requestId": "269",

                           "wat": {
                               'group_by_list': None,
                               "merge_product_list": None,
                               # "mergeProductId": [
                               #     {"xx1": ["AEMNRM01N.0B01", "AEMNE801N.0B01"]},
                               #     {"xx2": ["AGKNCE01N.0A01", "AFXNJ701N.0B01"]}],

                               "merge_prodg1_list": None,
                               "merge_operno_list": None
                           }
                           }

        final_result = CorrCompareWatBySiteAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                         config_dict=json_loads_dict.get("wat"), request_id="269")
        print("映射为all data schema:列数为{}".format(len(final_result.columns)))
        final_res = MergeAllDataSourceToResultTable.run([final_result])
        print("映射为all data schema:列数为{}".format(len(final_res.columns)))
        final_res.show()

        end = time.time()

        # remote spark time cost 65 s
        # no dropletions all 耗时
        print(f"remote spark  time cost: {end - start}s")

        return final_result


    inline_result = test_single_inline(spark)
    wat_result = test_single_wat(spark)

    result = MergeAllDataSourceToResultTable.run([inline_result, wat_result])
    assert len(result.columns) == 13, "result columns is not 13"
    result.show()

    # print(result1[['WEIGHT']].toPandas().describe())
    # MergeAllDataSourceToResultTable.run([result1]).toPandas().to_csv("result1.csv")

    # sum_col2 = final_res_.agg(F_sum("WEIGHT").alias("total_sum")).collect()[0]["total_sum"]
    # print(sum_col2)
    # time1 = datetime.now()
    # print(time1)
    # final_res_ = CorrCompareUvaByWaferAlgorithm.run(base_df=base_df_spark,
    #                                                  source_df=df_spark,
    #                                                 request_id='uva',
    #                                                 config_dict=json_loads_dict.get("uva")
    #                                                 )
    # time2 = datetime.now()
    # print(f"算法结果一共有{final_res_.count()}条")
    # print("算法结果写回数据库消耗的时间是：", time2 - time1)
    # final_res_.show()
    # # final_res_.show(30)
    # pprint(final_res_.toPandas().query("WEIGHT >0").head(1).to_json())
    # pprint(final_res_.columns)
    # # final_res_pandas = final_res_.toPandas()
    # # final_res_pandas.to_csv("final_res_pandas_big1.csv")
    # empty_df = final_res_.filter("1=0")
    # # 检测空表，None 输入
    # # MergeAllDataSourceToResultTable.run([empty_df, None]).show()
    #
    # MergeAllDataSourceToResultTable.run([final_res_]).show()
