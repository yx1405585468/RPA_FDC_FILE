import json
import pandas as pd
from src.inline.build_query import build_inline_query
# from src.inline.inline_by_site_test0410 import ExertInlineBySite
# from src.inline.inline_by_wafer_test0412 import ExertInlineByWafer
from src.inline.inline_bysite_algorithm import ExertInlineBySite
from src.inline.inline_bywafer_algorithm import ExertInlineByWafer

from src.utils import read_jdbc_executor
from pyspark.sql import SparkSession


def process_record_by_site(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = parse_JSON_config(
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
    print("good_site:")
    print(good_site)
    print("bad_site:")
    print(bad_site)

    query_sql = build_inline_query(parse_dict, properties_config)
    print(query_sql)
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    final_res = ExertInlineBySite.fit_by_site_model(df=doris_spark_df,
                                                    request_id=request_id,
                                                    merge_operno_list=merge_operno,
                                                    merge_prodg1_list=merge_prodg1,
                                                    merge_product_list=merge_product,
                                                    grpby_list=grpby_list,
                                                    good_site_columns=good_site,
                                                    bad_site_columns=bad_site)
    final_res.show(50)

    # 需要写的列
    # write_fields = ','.join(map(str, final_res.columns))
    # print(f"data column---{write_fields}")
    # final_res.write.format("doris") \
    #     .option("doris.table.identifier",
    #             f"{properties_config['doris_db']}.{properties_config['doris_inline_results_table']}") \
    #     .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
    #     .option("user", f"{properties_config['doris_user']}") \
    #     .option("password", f"{properties_config['doris_password']}") \
    #     .option("doris.write.fields", write_fields) \
    #     .option("doris.sink.batch.interval.ms", 50) \
    #     .option("doris.sink.task.use.repartition", True) \
    #     .save()


def process_record_by_wafer(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    # 解析JSON并且读取数据
    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site = parse_JSON_config(
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
    print("good_site:")
    print(good_site)
    print("bad_site:")
    print(bad_site)

    query_sql = build_inline_query(parse_dict, properties_config)
    print(query_sql)

    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    print("ExertInlineByWafer")
    final_res = ExertInlineByWafer.fit_by_wafer_model(df=doris_spark_df,
                                                      request_id=request_id,
                                                      grpby_list=grpby_list,
                                                      merge_operno_list=merge_operno,
                                                      merge_prodg1_list=merge_prodg1,
                                                      merge_product_list=merge_product)
    final_res.show(50)

    # 需要写的列
    # write_fields = ','.join(map(str, final_res.columns))
    # print(f"data column---{write_fields}")
    # final_res.write.format("doris") \
    #     .option("doris.table.identifier",
    #             f"{properties_config['doris_db']}.{properties_config['doris_inline_results_table']}") \
    #     .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
    #     .option("user", f"{properties_config['doris_user']}") \
    #     .option("password", f"{properties_config['doris_password']}") \
    #     .option("doris.write.fields", write_fields) \
    #     .option("doris.sink.batch.interval.ms", 50) \
    #     .option("doris.sink.task.use.repartition", True) \
    #     .save()


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

    # 获取good_site和bad_site
    try:
        good_site = list(parse_dict.get('goodSite')) if parse_dict.get('goodSite') else None
    except KeyError:
        good_site = None

    try:
        bad_site = list(parse_dict.get('badSite')) if parse_dict.get('badSite') else None
    except KeyError:
        bad_site = None

    # group by 子句中的字段
    group_by_list = parse_dict.get("groupByList")
    if group_by_list is None or len(group_by_list) == 0:
        group_by_list = ["PRODG1", "PRODUCT_ID", "OPER_NO"]
        # PRODUCT_ID, PROG1, CHAMBER 这3个存在一键合并的切换开关
        # 且一键合并PROG1时会自动一键合并PRODUCT_ID
        flag_merge_prodg1 = parse_dict.get('flagMergeAllProdg1')
        flag_merge_product_id = parse_dict.get('flagMergeAllProductId')
        flag_merge_chamber = parse_dict.get('flagMergeAllChamber')

        if flag_merge_prodg1 == '1':
            # 一键合并PROG1时，部分合并PROG1和PRODUCT_ID的情况都会被忽略
            merge_prodg1 = None
            merge_product = None
            group_by_list = ['OPER_NO']
        elif flag_merge_product_id == '1':
            # 一键合并PRODUCT_ID时，部分合并PRODUCT_ID的情况会被忽略
            merge_product = None
            group_by_list = ["PRODG1", "OPER_NO"]

    return parse_dict, request_id, group_by_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, good_site, bad_site


if __name__ == '__main__':
    import re
    import findspark

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
    # config_file_path = "D:/ikas-rca-job/tests/app_config.properties"
    # config_properties = spark_session.sparkContext.textFile(config_file_path).collect()
    # print(config_properties)
    # # 正则表达式匹配键值对，允许=号前后有空格
    # pattern = re.compile(r'\s*(.*?)\s*=\s*(.*)')
    # properties = {}
    # for line in config_properties:
    #     # 跳过空行和注释行
    #     if line.strip() == "" or line.strip().startswith("#"):
    #         continue
    #     match = pattern.match(line)
    #     if match:
    #         key, value = match.groups()
    #         properties[key] = value
    print("load_config_properties:")
    # print(properties)
    properties = {'rca_http_api_url': 'http://192.168.13.17:8089/internal/task/end', 'doris_ip': '192.168.13.229',
                  'doris_fe_http_port': '9030', 'doris_jdbc_url': 'jdbc:mysql://192.168.13.229:9030/rca',
                  'doris_user': 'root',
                  'doris_password': '123456', 'doris_db': 'rca', 'doris_fd_uva_table': 'DWD_FD_UVA_DATA',
                  'doris_inline_wafer_summary_table': 'DWD_INLINE_WAFER_SUMMARY',
                  'doris_uploaded_wafer_table': 'conf_wafer',
                  'doris_uva_results_table': 'uva_results', 'doris_inline_results_table': 'inline_results'}

    # Example: process_record_by_wafer, process_record_by_site

    # json_config_ = {"requestId": "269",
    #                 "algorithm": "inline_by_site",
    #                 "requestParam": {'dateRange': {'start': '2021-12-14 14:42:47', 'end': '2024-03-14 14:42:47'},
    #                                  'operNo': ['IKAS.OPEN.1'], 'uploadId': '288e7faf79604a498e3648e33581bff9',
    #                                  'goodSite': ['SITE2_VAL', 'SITE3_VAL', 'SITE1_VAL'],
    #                                  'badSite': ['SITE5_VAL', 'SITE4_VAL'],
    #                                  'flagMergeAllProdg1': '0',
    #                                  'flagMergeAllProductId': '0',
    #                                  'flagMergeAllChamber': '0',
    #                                  'mergeProdg1': [],
    #                                  'mergeProductId': [],
    #                                  'mergeEqp': [],
    #                                  'mergeChamber': [],
    #                                  'mergeOperno': []}
    #                 }
    json_config_ = {"requestId": "715", "algorithm": "inline_by_wafer",
                    "requestParam": {"dateRange": {"start": "2021-01-12 11:18:41", "end": "2024-04-12 11:18:41"},
                                     "uploadId": "8e521b3148ce4f4cab35651035f4be8b",
                                     # "goodSite": ["SITE16_VAL", "SITE7_VAL", "SITE11_VAL", "SITE14_VAL", "SITE17_VAL",
                                     #              "SITE20_VAL",
                                     #              "SITE9_VAL", "SITE12_VAL", "SITE13_VAL", "SITE15_VAL", "SITE18_VAL",
                                     #              "SITE19_VAL",
                                     #              "SITE10_VAL", "SITE8_VAL"],
                                     # "badSite": ["SITE6_VAL", "SITE2_VAL", "SITE3_VAL", "SITE5_VAL", "SITE1_VAL",
                                     #             "SITE4_VAL"],
                                     "goodSite": ["SITE16_VAL", "SITE7_VAL", "SITE11_VAL"],
                                     "badSite": ["SITE6_VAL", "SITE2_VAL", "SITE3_VAL", "SITE5_VAL"],
                                     "flagMergeAllProdg1": "0", "flagMergeAllProductId": "0",
                                     "flagMergeAllChamber": "0",
                                     "mergeProdg1": [], "mergeProductId": [], "mergeEqp": [], "mergeChamber": [],
                                     "mergeOperno": []}}

    # process_record_by_wafer(sparkSession=spark_session, json_config=json_config_, properties_config=properties)
    process_record_by_site(sparkSession=spark_session, json_config=json_config_, properties_config=properties)
