# -*- coding: utf-8 -*-
# time: 2024/6/20 17:12
# file: correlation_main.py
# author: waylee


import json
import logging

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from src.utils.score_executor import get_score_df
# from src.correlation.by_wafer_algorithm import CorrCompareByWaferAlgorithm
# 调用算法接口方式修改
from src.correlation.by_wafer_algorithms.main import CorrCompareByWaferAlgorithm
from src.correlation.by_site_algorithms.main import CorrCompareBySiteAlgorithm
from src.correlation.by_zone_algorithms.main import CorrCompareByZoneAlgorithm


from src.correlation.parse_json_to_config import ParseAlgorithmJsonConfig
from src.correlation.parse_json_to_sql import ParseAlgorithmSqlConfig
from src.correlation.by_zone_json_config import ByZoneConfigPare

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)


class AnomalyCharacterizationAlgorithm:
    """
    异常特征提取算法类，提供by wafer晶圆和by site站点处理数据两种方法。
    """
    @staticmethod
    def process_record_by_site(sparkSession: SparkSession, json_config: pd.DataFrame, properties_config: pd.DataFrame):
        """
        by wafer处理数据的方法。

        :param sparkSession: Spark会话，用于数据处理。
        :param json_config: JSON配置，包含算法配置信息。
        :param properties_config: 属性配置，包含数据库连接等配置信息。
        """

        query_sql_dict = ParseAlgorithmSqlConfig.parse_config(json.dumps(json_config),
                                                              properties_config=properties_config)
        all_arithmetic_configs = ParseAlgorithmJsonConfig.parse_config(json.dumps(json_config))

        if not query_sql_dict or not query_sql_dict.get("base_sql_query"):
            raise ValueError("base_sql_query 为空, 无法执行后续算法")

        try:
            # 读取Doris数据库中的数据
            base_query_df = sparkSession.read.format("jdbc") \
                .option("url", properties_config['doris_jdbc_url']) \
                .option("user", properties_config['doris_user']) \
                .option("password", properties_config['doris_password']) \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("query", query_sql_dict.get("base_sql_query")) \
                .option("numPartitions", "3") \
                .option("fetchsize", "1000") \
                .load()

            # 检查base_query_df是否为空
            if base_query_df.count() == 0:
                raise ValueError("base_query_df 数据集为空, 无法执行异常特征算法多数据集的关联分析...")
            base_query_df.persist(StorageLevel.MEMORY_AND_DISK)
            logger.info(f"样例数据集构建完成, 执行算法处理...")
            finally_res_df = CorrCompareBySiteAlgorithm(
                sparkSession=sparkSession,
                properties_config=properties_config,
                query_sql_dict=query_sql_dict,
                base_df=base_query_df,
                config_dict=all_arithmetic_configs).run()
            # 写入数据库
            score_df = get_score_df(spark_session=sparkSession, properties_config=properties_config, data_frame=finally_res_df)
            score_df.write.format("doris") \
                .option("doris.table.identifier",
                        f"{properties_config['doris_db']}.{properties_config['doris_correlation_results_table']}") \
                .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
                .option("user", f"{properties_config['doris_user']}") \
                .option("password", f"{properties_config['doris_password']}") \
                .option("doris.sink.batch.size", 10000) \
                .option("doris.sink.max-retries", 3) \
                .option("doris.sink.auto-redirect", True) \
                .save()
            score_df.show(10)
        except Exception as e:
            logger.error(f"异常特征算法处理数据发生异常: {e}")
            raise

    @staticmethod
    def process_record_by_wafer(sparkSession: SparkSession, json_config: pd.DataFrame, properties_config: pd.DataFrame):
        """
        by wafer处理数据的方法。

        :param sparkSession: Spark会话，用于数据处理。
        :param json_config: JSON配置，包含算法配置信息。
        :param properties_config: 属性配置，包含数据库连接等配置信息。
        """

        query_sql_dict = ParseAlgorithmSqlConfig.parse_config(json.dumps(json_config),
                                                              properties_config=properties_config)
        all_arithmetic_configs = ParseAlgorithmJsonConfig.parse_config(json.dumps(json_config))

        if not query_sql_dict or not query_sql_dict.get("base_sql_query"):
            raise ValueError("base_sql_query 为空, 无法执行后续算法")

        try:
            # 读取Doris数据库中的数据
            base_query_df = sparkSession.read.format("jdbc") \
                .option("url", properties_config['doris_jdbc_url']) \
                .option("user", properties_config['doris_user']) \
                .option("password", properties_config['doris_password']) \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("query", query_sql_dict.get("base_sql_query")) \
                .option("numPartitions", "3") \
                .option("fetchsize", "1000") \
                .load()

            # 检查base_query_df是否为空
            if base_query_df.count() == 0:
                raise ValueError("base_query_df 数据集为空, 无法执行异常特征算法多数据集的关联分析...")
            base_query_df.persist(StorageLevel.MEMORY_AND_DISK)
            logger.info(f"样例数据集构建完成, 执行算法处理...")
            finally_res_df = CorrCompareByWaferAlgorithm(
                sparkSession=sparkSession,
                properties_config=properties_config,
                query_sql_dict=query_sql_dict,
                base_df=base_query_df,
                config_dict=all_arithmetic_configs).run()
            # 写入数据库
            score_df = get_score_df(spark_session=sparkSession, properties_config=properties_config, data_frame=finally_res_df)
            score_df.write.format("doris") \
                .option("doris.table.identifier",
                        f"{properties_config['doris_db']}.{properties_config['doris_correlation_results_table']}") \
                .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
                .option("user", f"{properties_config['doris_user']}") \
                .option("password", f"{properties_config['doris_password']}") \
                .option("doris.sink.batch.size", 10000) \
                .option("doris.sink.max-retries", 3) \
                .option("doris.sink.auto-redirect", True) \
                .save()
            score_df.show(10)
        except Exception as e:
            logger.error(f"异常特征算法处理数据发生异常: {e}")
            raise

    @staticmethod
    def process_record_by_zone(sparkSession: SparkSession, json_config: pd.DataFrame, properties_config: pd.DataFrame):
        """
        by wafer处理数据的方法。

        :param sparkSession: Spark会话，用于数据处理。
        :param json_config: JSON配置，包含算法配置信息。
        :param properties_config: 属性配置，包含数据库连接等配置信息。
        """

        query_sql_dict = ParseAlgorithmSqlConfig.parse_config(json.dumps(json_config),
                                                              properties_config=properties_config)
        all_arithmetic_configs = ParseAlgorithmJsonConfig.parse_config(json.dumps(json_config))
        json_data = json.loads(json.dumps(json_config))
        request_param = json_data.get("requestParam", {})
        upload_id = request_param.get("uploadId")
        db_config = {
            'host': properties_config['doris_ip'],
            'port': 9030,
            'user': properties_config['doris_user'],
            'password': properties_config['doris_password'],
            'database': properties_config['doris_db']
        }
        query = f"SELECT NAME, MODE, SITES, ANALYSIS, LABEL FROM rca.CONF_MODE WHERE UPLOAD_ID='{upload_id}'"

        mysql_to_json = ByZoneConfigPare(**db_config)
        mode_info_json = mysql_to_json.run(query)

        final_configs = ParseAlgorithmJsonConfig.merge_mode_info(mode_info_json, all_arithmetic_configs)
        
        if not query_sql_dict or not query_sql_dict.get("base_sql_query"):
            raise ValueError("base_sql_query 为空, 无法执行后续算法")

        try:
            # 读取Doris数据库中的数据
            base_query_df = sparkSession.read.format("jdbc") \
                .option("url", properties_config['doris_jdbc_url']) \
                .option("user", properties_config['doris_user']) \
                .option("password", properties_config['doris_password']) \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("query", query_sql_dict.get("base_sql_query")) \
                .option("numPartitions", "3") \
                .option("fetchsize", "1000") \
                .load()

            # 检查base_query_df是否为空
            if base_query_df.count() == 0:
                raise ValueError("base_query_df 数据集为空, 无法执行异常特征算法多数据集的关联分析...")
            base_query_df.persist(StorageLevel.MEMORY_AND_DISK)
            logger.info(f"样例数据集构建完成, 执行算法处理...")
            finally_res_df = CorrCompareByZoneAlgorithm(
                sparkSession=sparkSession,
                properties_config=properties_config,
                query_sql_dict=query_sql_dict,
                base_df=base_query_df,
                config_dict=final_configs).run()
            # 写入数据库
            score_df = get_score_df(spark_session=sparkSession, properties_config=properties_config, data_frame=finally_res_df)
            score_df.write.format("doris") \
                .option("doris.table.identifier",
                        f"{properties_config['doris_db']}.{properties_config['doris_correlation_results_table']}") \
                .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
                .option("user", f"{properties_config['doris_user']}") \
                .option("password", f"{properties_config['doris_password']}") \
                .option("doris.sink.batch.size", 10000) \
                .option("doris.sink.max-retries", 3) \
                .option("doris.sink.auto-redirect", True) \
                .save()
            score_df.show(10)
        except Exception as e:
            logger.error(f"异常特征算法处理数据发生异常: {e}")
            raise
