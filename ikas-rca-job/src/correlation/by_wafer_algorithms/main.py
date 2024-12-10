from src.correlation.building_dataframe import BuildSparkDataframe
from src.correlation.by_wafer_algorithms.compare_inline import CorrCompareInlineByWaferAlgorithm
from src.correlation.by_wafer_algorithms.compare_uva import CorrCompareUvaByWaferAlgorithm
from src.correlation.by_wafer_algorithms.compare_wat import CorrCompareWatByWaferAlgorithm
from src.correlation.by_wafer_algorithms.compare_process import CorrCompareProcessByWaferAlgorithm
import pyspark
from pyspark.sql import SparkSession
import pandas as pd
from src.correlation.common_process.data_preprocessing import MergeAllDataSourceToResultTable


class CorrCompareByWaferAlgorithm(object):
    algo_dispatch_mapt = {
        'uva': CorrCompareUvaByWaferAlgorithm,
        'inline': CorrCompareInlineByWaferAlgorithm,
        'wat': CorrCompareWatByWaferAlgorithm,
        'process': CorrCompareProcessByWaferAlgorithm,
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
            "------------------------------------- by wafer 算法执行入参配置如下 -------------------------------------")
        import json
        # print(f"properties_config: {self.properties_config.to_json()}")
        print(f"query_sql_dict: {json.dumps(self.query_sql_dict)}")
        print(f"config_dict: {json.dumps(self.config_dict)}")
        for key, value in self.config_dict.items():
            if key == 'request_id':
                continue

            elif key in ['uva', 'inline', 'wat', 'process']:
                if self.config_dict.get(key) is not None:
                    source_df = BuildSparkDataframe.get_dataframe(sparkSession=self.sparkSession,
                                                                  query_sql_dict=self.query_sql_dict,
                                                                  algo_type=key,
                                                                  properties_config=self.properties_config)
                    # 获取数据源对应处理的算法类
                    CurrrentCorrByWaferAlgorithm = self.algo_dispatch_mapt[key]

                    if source_df:
                        # 执行uva关系检测算法
                        result = CurrrentCorrByWaferAlgorithm.run(
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

    def test_single_uva(spark):
        from pprint import pprint
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1_1.csv")

        pprint(df_pandas.columns)

        # 异常表征 df
        base_df = df_pandas.query(
            "PRODG1 == 'L11CB14A' & PRODUCT_ID == 'AMKNS301N.0A01'"
            " & OPE_NO == '1F.EEK10' & EQP_NAME =='EKT72' & CHAMBER_NAME =='EKT72_PM1'"
            " & PARAMETRIC_NAME =='APC_POSITION#AOTU_STEP_2#MEAN'"

        )

        # df_spark = ps.from_pandas(base_df).to_spark()

        base_df['ParName'] = np.random.normal(size=len(base_df))
        base_df_spark = ps.from_pandas(base_df).to_spark().select("WAFER_ID", col("RESULT").alias("ParName"))
        # base_df_spark.show()

        df_spark = ps.from_pandas(df_pandas).to_spark()
        # print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        # df_spark.show()

        json_loads_dict = {
            "request_id": "uva",
            "uva": {
                "group_by_list": ['PRODG1', 'PRODUCT_ID', 'OPE_NO', 'EQP_NAME', "CHAMBER_NAME"],
                "merge_prodg1_list": None,
                "merge_product_list": [{"xx_cc": ["AFGNK401N.0A01", "AFGN1501N.0C02"]}],
                # "mergeProductId": None,
                "merge_eqp_list": [],
                "merge_chamber_list": [],
                "merge_operno_list": None,
                # 'mergeOperno': [{"2F.CDS10_XX.TDS01": ["2F.CDS10", "XX.TDS01"]},
                #                 {"2F.CDS20_XX.CDS20": ["2F.CDS20", "XX.CDS20"]}]
            }
        }

        final_res_ = CorrCompareByWaferAlgorithm(base_df=base_df_spark,
                                                 source_df=df_spark,
                                                 config_dict=json_loads_dict
                                                 ).run()

        final_res_.show()
        return final_res_


    def test_single_inline(spark):
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\inline_case5_label_1.csv")

        df_spark = ps.from_pandas(df_pandas).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        uva_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\DWD_POC_CASE_FD_UVA_DATA_CASE1_PROCESSED1_1.csv")

        # 异常表征 df,from uva  => compare inline
        base_df = df_pandas.query(
            "OPE_NO == '1C.CDG10' & PARAMETRIC_NAME == 'EEW0'"
        )

        base_df_spark = ps.from_pandas(
            base_df[['WAFER_ID', 'AVERAGE']].rename(columns={'AVERAGE': 'ParName'})).to_spark()
        base_df_spark.show()
        #
        # join_result = df_pandas.merge(base_df[['WAFER_ID', 'AVERAGE']].rename(columns={'AVERAGE': 'ParName'}), on=['WAFER_ID'], how='left')
        #
        # # print("join 结果\n", join_result.groupby(['OPE_NO', 'PARAMETRIC_NAME_x'])['AVERAGE'].agg(lambda x: x.isna().sum()))
        # pprint(join_result[["AVERAGE", 'ParName', "WAFER_ID"]])

        # exit(0)

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
                           }
                           }

        final_result = CorrCompareInlineByWaferAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
                                                             config_dict=json_loads_dict.get("inline"),
                                                             request_id="269")
        #
        # pprint(df_spark.dtypes)
        final_result.show()

        return final_result


    def test_single_wat(spark):
        import time

        start = time.time()
        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\wat_select.csv")
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(
        #     ["AEMNRM01N.0B01", "AEMNE801N.0B01", "AFXNE001N.0C01", "AGKNCE01N.0A01", "AFXNJ701N.0B01"])]
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(["AEMNRM01N.0B01", "AEMNE801N.0B01"])]
        df_spark = ps.from_pandas(df_pandas).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()

        base_df = df_pandas.query("""
            OPE_NO == 'ST.TTS10' & PARAMETRIC_NAME == 'BVjLN+LN+_I_p14_10nA' & PRODUCT_ID == 'AEMNRM01N.0B01'
            """)[['AVERAGE', "WAFER_ID"]].rename(columns={"AVERAGE": "ParName"})

        base_df_spark = ps.from_pandas(base_df).to_spark()
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

        final_result = CorrCompareWatByWaferAlgorithm.run(base_df=base_df_spark, source_df=df_spark,
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


    # result = test_single_wat(spark)
    result1 = test_single_wat(spark)

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
