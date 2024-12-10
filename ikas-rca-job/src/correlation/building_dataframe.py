# -*- coding: utf-8 -*-
# time: 2024/6/22 11:57
# file: building_dataframe.py
# author: waylee
import logging
from typing import Optional

import pandas as pd
import pyspark.sql
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


class BuildSparkDataframe:

    @staticmethod
    def get_dataframe(sparkSession: SparkSession, algo_type: str, query_sql_dict: dict,
                      properties_config: pd.DataFrame) -> Optional[pyspark.sql.DataFrame]:
        """
        从Doris数据库中读取数据并构建Spark DataFrame。

        :param sparkSession: Spark会话，用于数据处理。
        :param algo_type: 算法类型，用于从query_sql_dict中获取对应的SQL查询。
        :param query_sql_dict: 包含不同算法类型的SQL查询。
        :param properties_config: 包含数据库连接配置。
        :return: 如果成功读取数据，返回Spark DataFrame；如果数据集为空或发生错误，返回None。
        """
        logger.info(f"获取{algo_type}算法数据集...")

        if algo_type not in query_sql_dict:
            logger.error(f"query_sql_dict中不存在指定的算法类型 {algo_type}, {query_sql_dict.keys()}")
            return None

        try:
            sparkSession.sparkContext.setLogLevel("info")
            result_df = sparkSession.read.format("jdbc") \
                .option("url", properties_config['doris_jdbc_url']) \
                .option("user", properties_config['doris_user']) \
                .option("password", properties_config['doris_password']) \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("query", query_sql_dict[algo_type]) \
                .option("numPartitions", "3") \
                .option("fetchsize", "1000") \
                .load()

            # 检查数据集是否为空
            if result_df.count() == 0:
                logger.error(f"result_df for query {algo_type} 数据集为空。")
                return None

            num_partitions = min(int(result_df.count() ** 0.5), 1000)
            result_df.show(3)
            return result_df.repartition(num_partitions, "WAFER_ID")

        except Exception as e:
            logger.error(f"构建spark dataframe数据集时异常: {e}")
            raise e
