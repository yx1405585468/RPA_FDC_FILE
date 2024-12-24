# encoding:utf-8

import pyspark
from pyspark.sql import SparkSession
from src.correlation.by_site_algorithms.compare_wat import CorrCompareWatBySiteAlgorithm
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from src.correlation.common_process.data_preprocessing import (
    compute_not_na_mean_by_row_func,
)


class CorrCompareWatByZoneAlgorithm(CorrCompareWatBySiteAlgorithm):

    @staticmethod
    def run(
        base_df: pyspark.sql.DataFrame,
        source_df: pyspark.sql.DataFrame,
        request_id: str,
        config_dict: dict,
    ) -> pyspark.sql.DataFrame:

        # 获取by zone 参数
        def reshape_site_into_zone_by_row_mean(
            source_df: pyspark.sql.DataFrame,
        ) -> pyspark.sql.DataFrame:
            mode_info = config_dict.get("mode_info")
            assert mode_info is not None, "By zone 参数 mode_info 丢失"
            zone_info = mode_info.get("NORMAL")
            assert zone_info is not None, "By zone 参数 NORMAL 丢失"

            # 获取 base_df 中NAME 列的zone value, 因为zone列会存在空值，所以，需要过滤掉空值

            # 获取spark context 确保pandas udf可用
            all_site_list = []
            map_zone_by_row_mean_transformer = {}
            # 处理source_df 数据,by zone 聚合后，一个zone 成为一列,当做新的site, 调用by site 算法
            for zone_name, site_list in zone_info.items():
                map_zone_by_row_mean_transformer[zone_name] = (
                    compute_not_na_mean_by_row_func(site_list)
                )
                all_site_list.extend(site_list)

            # 过滤列，计算列检查类型，如果是指定类型，不用转换，否则转换
            source_df = source_df.withColumns(
                {
                    col: F.col(col).cast("double")
                    for col in all_site_list
                    if source_df.schema[col].dataType != DoubleType()
                }
            )

            source_df = source_df.withColumns(map_zone_by_row_mean_transformer)
            return source_df

        source_df = reshape_site_into_zone_by_row_mean(source_df)
        source_df.show()

        # 调用父类的静态方法，把聚合后的zone 当做site对待，调用by site 算法
        return super(CorrCompareWatByZoneAlgorithm, CorrCompareWatByZoneAlgorithm).run(
            base_df=base_df,
            source_df=source_df,
            request_id=request_id,
            config_dict=config_dict,
        )


if __name__ == "__main__":
    import time
    import findspark
    import pyspark.pandas as ps
    from src.correlation.common_process.data_preprocessing import (
        MergeAllDataSourceToResultTable,
    )
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")

    def get_local_spark():

        spark_home = r"D:\software\spark-3.3.0-bin-hadoop3"
        python_path = r"D:\software\Anaconda3\envs\python39\python.exe"
        findspark.init(spark_home, python_path)
        spark = (
            SparkSession.builder.appName("example")
            .config("master", "local[4]")
            .getOrCreate()
        )

        return spark

    def timer(func):

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} 耗时 {end - start} seconds")
            return result

        return wrapper

    @timer
    def test_single_wat(spark: pyspark.sql.SparkSession):

        df_pandas = pd.read_csv(
            r"D:\xxs_project\2024\RCA根因分析\test_data\wat_select.csv"
        )
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(
        #     ["AEMNRM01N.0B01", "AEMNE801N.0B01", "AFXNE001N.0C01", "AGKNCE01N.0A01", "AFXNJ701N.0B01"])]
        # df_pandas = df_pandas[df_pandas['PRODUCT_ID'].isin(["AEMNRM01N.0B01", "AEMNE801N.0B01"])]
        base_df = df_pandas.query(
            """
            OPE_NO == 'ST.TTS10' & PARAMETRIC_NAME == 'BVjLN+LN+_I_p14_10nA' & PRODUCT_ID == 'AEMNRM01N.0B01'
            """
        )

        # 根据wafer id 过滤
        df_spark = ps.from_pandas(
            df_pandas[df_pandas["WAFER_ID"].isin(base_df["WAFER_ID"].unique())]
        ).to_spark()
        print(f"df_spark shape: ({df_spark.count()}, {len(df_spark.columns)})")
        df_spark.show()
        print(base_df)
        select_site_cols = []
        for col_name in base_df.columns:
            if col_name.startswith("SITE") and col_name.endswith("VAL"):
                select_site_cols.append(col_name)

        from pprint import pprint

        pprint(select_site_cols)

        zone_info_dict = {
            "NORMAL": {
                "zone1": ["SITE{}_VAL".format(i) for i in range(1, 4)],
                "zone2": ["SITE{}_VAL".format(i) for i in range(4, 7)],
                "zone3": ["SITE{}_VAL".format(i) for i in range(7, 10)],
                "zone4": ["SITE{}_VAL".format(i) for i in range(10, 13)],
                "zone5": ["SITE{}_VAL".format(i) for i in range(13, 16)],
            }
        }

        # 构建异常表征数据
        base_df_spark = ps.from_pandas(
            pd.DataFrame()
            .assign(
                **{
                    key: base_df.filter(items=value).apply(
                        lambda x: x.mean(), axis="columns"
                    )
                    for key, value in zone_info_dict.get("NORMAL").items()
                }
            )
            .apply(lambda x: x.mean(), axis="index")
            .reset_index()
            .rename(columns={"index": "NAME", 0: "value"})
        ).to_spark()

        base_df_spark.show()

        json_loads_dict = {
            "requestId": "269",
            "wat": {
                "group_by_list": [
                    "PRODG1",
                    "PRODUCT_ID",
                ],
                "merge_product_list": None,
                # "mergeProductId": [
                #     {"xx1": ["AEMNRM01N.0B01", "AEMNE801N.0B01"]},
                #     {"xx2": ["AGKNCE01N.0A01", "AFXNJ701N.0B01"]}],
                "merge_prodg1_list": None,
                "merge_operno_list": None,
                "mode_info": zone_info_dict,
            },
        }

        final_result = CorrCompareWatByZoneAlgorithm.run(
            base_df=base_df_spark,
            source_df=df_spark,
            config_dict=json_loads_dict.get("wat"),
            request_id="269",
        )

        final_result.show()
        print("映射为all data schema前:列数为{}".format(len(final_result.columns)))
        final_res = MergeAllDataSourceToResultTable.run([final_result])
        print("映射为all data schema 后:列数为{}".format(len(final_res.columns)))
        final_result = final_res.orderBy(F.col("WEIGHT").desc(), F.col("COUNT").desc())

        # remote spark time cost 65 s
        # no dropletions all 耗时

        return final_result

    # 传入spark Session
    spark = get_local_spark()
    # 测试函数耗时
    result = test_single_wat(spark)
    result.show()
