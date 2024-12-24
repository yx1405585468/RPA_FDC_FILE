import argparse
import json
import logging
import traceback
from datetime import datetime

from pyspark.sql import SparkSession

from src.correlation.correlation_main import AnomalyCharacterizationAlgorithm
from src.defect import defect_main
from src.exceptions.rca_base_exception import RCABaseException
from src.inline import inline_main
from src.fdc_advanced import fdc_advanced_main
from src.correlation import correlation_main
from src.utils.http_utils import notify_http_api
from src.uva import uva_main
import re

from src.wat import wat_main


def create_spark_session() -> SparkSession:
    """
    创建Spark会话
    """
    return SparkSession.builder \
        .appName("RootCauseAnalysisPYSparkJob") \
        .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
        .getOrCreate()


def execute_algorithm(sparkSession, json_config, properties_config):
    """
    根据算法类型执行不同的ETL算法
    """
    if json_config['algorithm'] == "uva":
        # 执行 uva 算法的 ETL
        uva_main.process_record(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "inline_by_wafer":
        # 执行 inline 算法的 ETL
        inline_main.process_record_by_wafer(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "inline_by_site":
        # 执行 inline 算法的 ETL
        inline_main.process_record_by_site(sparkSession, json_config, properties_config)
    elif json_config['algorithm'].startswith("inline_by"):
        # by mode异常特征提取算法
        inline_main.process_record_by_zone(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "wat_by_wafer":
        # 执行 wat 算法的 ETL
        wat_main.process_record_by_wafer(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "wat_by_site":
        # 执行 wat 算法的 ETL
        wat_main.process_record_by_site(sparkSession, json_config, properties_config)
    elif json_config['algorithm'].startswith("wat_by"):
        # 执行 wat 算法的 ETL
        wat_main.process_record_by_zone(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "fdc_advanced":
        # 执行 inline 算法的 ETL
        fdc_advanced_main.process_record_fdc_advanced(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "defect":
        # 执行 defect 算法的 ETL
        defect_main.process_record_defect(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "correlation_by_wafer":
        # by wafer异常特征提取算法
        AnomalyCharacterizationAlgorithm.process_record_by_wafer(sparkSession, json_config, properties_config)
    elif json_config['algorithm'] == "correlation_by_site":
        # by site异常特征提取算法
        AnomalyCharacterizationAlgorithm.process_record_by_site(sparkSession, json_config, properties_config)
    elif json_config['algorithm'].startswith("correlation_by"):
        # by mode异常特征提取算法
        AnomalyCharacterizationAlgorithm.process_record_by_zone(sparkSession, json_config, properties_config)



def main():
    # 初始化Spark会话
    spark_session = create_spark_session()
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='Process JSON file')
    parser.add_argument('--json_file', type=str, help='Path to JSON file')
    args = parser.parse_args()
    # 读取传入的JSON配置
    if args.json_file:
        json_file = spark_session.sparkContext.textFile(args.json_file).collect()
        json_config = json.loads(json_file[0])
        ope_nos = json_config.get('requestParam', {}).get('operNo')
        uva_key = json_config.get('requestParam', {}).get('uva')
        inline_key = json_config.get('requestParam', {}).get('inline')
        wat_key = json_config.get('requestParam', {}).get('wat')
        process_key = json_config.get('requestParam', {}).get('process')
        if uva_key:
            operno_list = uva_key.get("operNo")
            uva_list = []
            for i in operno_list:
                uva_list.extend(i.split(","))
            json_config['requestParam']['uva']["operNo"] = uva_list
        if inline_key:
            operno_list = inline_key.get("operNo")
            inline_list = []
            for i in operno_list:
                inline_list.extend(i.split(","))
            json_config['requestParam']['inline']["operNo"] = inline_list
        if wat_key:
            operno_list = wat_key.get("operNo")
            wat_list = []
            for i in operno_list:
                wat_list.extend(i.split(","))
            json_config['requestParam']['wat']["operNo"] = wat_list
        if process_key:
            operno_list = process_key.get("operNo")
            process_list = []
            for i in operno_list:
                process_list.extend(i.split(","))
            json_config['requestParam']['process']["operNo"] = process_list
        # 处理dyb类型站点
        if ope_nos:
            data = []
            for single_ope in ope_nos:
                data.extend(single_ope.split(','))
            json_config['requestParam']['operNo'] = data
        logger.info(f"json_config:{json_config}")
    try:
        # 读取Spark配置
        config_file_path = "hdfs:///ikas_rca/rca_config.properties"
        config_properties = spark_session.sparkContext.textFile(config_file_path).collect()
        # 正则表达式匹配键值对，允许=号前后有空格
        pattern = re.compile(r'\s*(.*?)\s*=\s*(.*)')

        properties = {}
        for line in config_properties:
            # 跳过空行和注释行
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            match = pattern.match(line)
            if match:
                key, value = match.groups()
                properties[key] = value
        print("load_config_properties:")
        print(properties)

        # 执行业务算法
        execute_algorithm(spark_session, json_config, properties)

        # 执行完成通知HTTP接口
        task_id = json_config['requestId']
        task_status = "SUCCESS"  # 默认成功
        msg = None  # 默认为null
        end_time = int(datetime.now().timestamp() * 1000)  # 当前时间的毫秒数
        notify_http_api(task_id, task_status, msg, end_time, properties)
    except RCABaseException as rca:
        # 处理异常
        task_id = json_config.get('requestId', 'unknown_task_id')
        task_status = "FAILED"
        end_time = int(datetime.now().timestamp() * 1000)
        print(f"------task algorithm error-----: {rca.message}")
        notify_http_api(task_id, task_status, rca.message, end_time, properties)
        raise rca
    except Exception as e:
        # 处理异常
        task_id = json_config.get('requestId', 'unknown_task_id')
        task_status = "FAILED"
        end_time = int(datetime.now().timestamp() * 1000)
        print(f"------task spark exception error-----: {str(e)}")
        exc_info = traceback.format_exc()
        print(f"------exc info-----: {exc_info}")
        notify_http_api(task_id, task_status, f"spark exception: {e}", end_time, properties)
        raise e
    finally:
        # 关闭Spark会话
        spark_session.stop()


if __name__ == "__main__":
    main()
