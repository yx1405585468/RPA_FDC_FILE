import json
import pandas as pd
from src.utils import read_jdbc_executor
from src.fdc_advanced.fdc_advanced_algorithm import ExertAdvancedAlgorithm

# build_query根据run data表名改写
from src.uva.build_query import build_uva_query


def process_record_fdc_advanced(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    # 解析JSON并且读取数据
    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber = parse_JSON_config(
        df_info_)
    print("parse_dict:")
    print(parse_dict)
    print("request_id:")
    print(request_id)
    print("grpby_list:")
    print(grpby_list)
    print("merge_operno:")
    print(merge_operno)
    print("merge_prodg1:")
    print(merge_prodg1)
    print("merge_product:")
    print(merge_product)
    print("merge_eqp:")
    print(merge_eqp)
    print("merge_chamber:")
    print(merge_chamber)

    query_sql = build_uva_query(parse_dict, properties_config)
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    doris_spark_df.persist()
    result = ExertAdvancedAlgorithm.fit_advanced_model(df=doris_spark_df,
                                                       grpby_list=grpby_list,
                                                       merge_operno_list=merge_operno,
                                                       merge_prodg1_list=merge_prodg1,
                                                       merge_product_list=merge_product,
                                                       merge_eqp_list=merge_eqp,
                                                       merge_chamber_list=merge_chamber,
                                                       request_id=request_id)
    # result.show()
    # 需要写的列
    write_fields = ','.join(map(str, result.columns))
    print(f"data---{write_fields}")
    result.write.format("doris") \
        .option("doris.table.identifier",
                f"{properties_config['doris_db']}.{properties_config['doris_fdc_advanced_results_table']}") \
        .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
        .option("user", f"{properties_config['doris_user']}") \
        .option("password", f"{properties_config['doris_password']}") \
        .option("doris.write.fields", write_fields) \
        .option("doris.sink.batch.interval.ms", 50) \
        .option("doris.sink.batch.size", 100000) \
        .option("doris.sink.max-retries", 3) \
        .option("doris.sink.auto-redirect", True) \
        .option("doris.sink.task.use.repartition", True) \
        .save()


def parse_JSON_config(df: pd.DataFrame):
    request_id = df["requestId"].values[0]
    request_params = df["requestParam"].values[0]
    parse_dict = json.loads(request_params)

    # PRODUCT_ID, PROG1, EQP, CHAMBER, OPER_NO存在部分合并的情况
    try:
        # OPER_NO的部分合并结果
        merge_operno = list(parse_dict.get('mergeOperno')) if parse_dict.get('mergeOperno') else None
    except KeyError:
        merge_operno = None

    try:
        # PROG1的部分合并结果
        merge_prodg1 = list(parse_dict.get('mergeProdg1')) if parse_dict.get('mergeProdg1') else None
    except KeyError:
        merge_prodg1 = None

    try:
        # PRODUCT_ID的部分合并结果
        merge_product = list(parse_dict.get('mergeProductId')) if parse_dict.get('mergeProductId') else None
    except KeyError:
        merge_product = None

    try:
        # EQP的部分合并结果
        merge_eqp = list(parse_dict.get('mergeEqp')) if parse_dict.get('mergeEqp') else None
    except KeyError:
        merge_eqp = None

    try:
        # CHAMBER的部分合并结果
        merge_chamber = list(parse_dict.get('mergeChamber')) if parse_dict.get('mergeChamber') else None
    except KeyError:
        merge_chamber = None

    # group by 子句中的字段
    group_by_list = parse_dict.get("groupByList")
    if group_by_list is None or len(group_by_list) == 0:
        group_by_list = ["PRODG1", "PRODUCT_ID", "OPER_NO", "EQP_NAME", "TOOL_NAME"]
        # PRODUCT_ID, PROG1, CHAMBER 这3个存在一键合并的切换开关
        # 且一键合并PROG1时会自动一键合并PRODUCT_ID
        flag_merge_prodg1 = parse_dict.get('flagMergeAllProdg1')
        flag_merge_product_id = parse_dict.get('flagMergeAllProductId')
        flag_merge_chamber = parse_dict.get('flagMergeAllChamber')

        if flag_merge_prodg1 == '1':
            # 一键合并PROG1时，部分合并PROG1和PRODUCT_ID的情况都会被忽略
            merge_prodg1 = None
            merge_product = None
            group_by_list = ['OPER_NO', "EQP_NAME", 'TOOL_NAME']
            if flag_merge_chamber == '1':
                group_by_list = ['OPER_NO', "EQP_NAME"]
        elif flag_merge_product_id == '1':
            # 一键合并PRODUCT_ID时，部分合并PRODUCT_ID的情况会被忽略
            merge_product = None
            group_by_list = ["PRODG1", "OPER_NO", "EQP_NAME", "TOOL_NAME"]
            if flag_merge_chamber == '1':
                # 一键合并CHAMBER时，部分合并CHAMBER的情况会被忽略
                group_by_list = ["PRODG1", 'OPER_NO', "EQP_NAME"]
        elif flag_merge_chamber == '1':
            merge_chamber = None
            group_by_list = ["PRODG1", "PRODUCT_ID", "OPER_NO", "EQP_NAME"]

    return parse_dict, request_id, group_by_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber


if __name__ == '__main__':
    import findspark
    import re
    from pyspark.sql import SparkSession

    findspark.init()


    def create_spark_session() -> SparkSession:
        """
        创建Spark会话
        """
        return SparkSession.builder \
            .appName("RootCauseAnalysisPYSparkJob") \
            .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
            .getOrCreate()


    spark_session = create_spark_session()

    # 读取Spark配置
    config_file_path = "D:/ikas-rca-job/tests/app_config.properties"
    config_properties = spark_session.sparkContext.textFile(config_file_path).collect()
    print(config_properties)
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

    json_config_ = {"requestId": "267",
                    "algorithm": "fdc_advanced",
                    "requestParam": {"dateRange": {"start": "2021-12-06 17:42:19",
                                                   "end": "2024-03-06 17:42:19"},
                                     "operNo": [],
                                     "eqp": [],
                                     "recipeName": [],
                                     "uploadId": "str",
                                     "flagMergeAllProdg1": "1",
                                     "flagMergeAllProductId": "0",
                                     "flagMergeAllChamber": "1",
                                     "mergeProdg1": [],
                                     "mergeProductId": [],
                                     "mergeEqp": [],
                                     "mergeChamber": [],
                                     "mergeOperno": [{'kl': ['oper4M.EFM0', 'oper4M.EFM1']}]
                                     }
                    }

    process_record_fdc_advanced(sparkSession=spark_session, json_config=json_config_, properties_config=properties)
