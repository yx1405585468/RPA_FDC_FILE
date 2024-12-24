from src.correlation.building_dataframe import BuildSparkDataframe
from src.correlation.by_zone_algorithms.compare_inline import (
    CorrCompareInlineByZoneAlgorithm,
)
from src.correlation.by_zone_algorithms.compare_wat import CorrCompareWatByZoneAlgorithm
import pyspark
from pyspark.sql import SparkSession
import pandas as pd
from src.correlation.common_process.data_preprocessing import (
    MergeAllDataSourceToResultTable,
)


class CorrCompareByZoneAlgorithm(object):
    algo_dispatch_map = {
        "inline": CorrCompareInlineByZoneAlgorithm,
        "wat": CorrCompareWatByZoneAlgorithm,
    }

    def __init__(
        self,
        sparkSession: SparkSession,
        properties_config: pd.DataFrame,
        query_sql_dict: dict,
        base_df: pyspark.sql.DataFrame,
        config_dict: dict,
    ):
        self.sparkSession = sparkSession
        self.properties_config = properties_config
        self.query_sql_dict = query_sql_dict
        self.config_dict = config_dict
        self.base_df = base_df

    def run(self):
        request_id = self.config_dict["request_id"]
        result_df_list = []

        print(
            "------------------------------------- by zone 算法执行入参配置如下 -------------------------------------"
        )
        # print(f"properties_config: {self.properties_config.to_json()}")
        import json

        print(f"query_sql_dict: {json.dumps(self.query_sql_dict)}")
        print(f"config_dict: {json.dumps(self.config_dict)}")

        for key, value in self.config_dict.items():
            if key == "request_id":
                continue

            elif key in ["inline", "wat"]:
                if self.config_dict.get(key) is not None:
                    source_df = BuildSparkDataframe.get_dataframe(
                        sparkSession=self.sparkSession,
                        query_sql_dict=self.query_sql_dict,
                        algo_type=key,
                        properties_config=self.properties_config,
                    )
                    # 获取数据源对应处理的算法类
                    current_corr_by_zone_algorithm = self.algo_dispatch_map[key]

                    if source_df:
                        # 执行byzone关系检测算法
                        result = current_corr_by_zone_algorithm.run(
                            base_df=self.base_df,
                            source_df=source_df,
                            config_dict=self.config_dict.get(key),
                            request_id=request_id,
                        )

                        result_df_list.append(result)

        # 各个数据源处理的结果后处理
        return MergeAllDataSourceToResultTable().run(result_df_list)


if __name__ == "__main__":
    pass
