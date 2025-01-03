import json
import pandas as pd
from datetime import datetime

from pyspark import StorageLevel

from src.wat.build_query import build_wat_query
from src.wat.wat_bywafer_algorithm import ExertWatByWafer
from src.wat.wat_bysite_algorithm import ExertWatBySite
from src.wat.wat_byzone_algorithm import ExertWatByZone
from src.utils import read_jdbc_executor
from src.correlation.by_zone_json_config import ByZoneConfigPare


def process_record_by_wafer(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_parametric_name, good_site, bad_site = parse_JSON_config(
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
    print("merge_parametric_name:")
    print(merge_parametric_name)
    print("good_site:")
    print(good_site)
    print("bad_site:")
    print(bad_site)

    query_sql = build_wat_query(parse_dict, properties_config)
    print(query_sql)

    time1 = datetime.now()
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    time2 = datetime.now()
    print("从数据库中获取数据消耗的时间是：", time2 - time1)

    final_res = ExertWatByWafer.fit_by_wafer_model(df=doris_spark_df,
                                                   request_id=request_id,
                                                   grpby_list=grpby_list,
                                                   merge_operno_list=merge_operno,
                                                   merge_prodg1_list=merge_prodg1,
                                                   merge_product_list=merge_product,
                                                   merge_parametric_name_list=merge_parametric_name)
    time3 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time3 - time2)

    # 需要写的列
    write_fields = ','.join(map(str, final_res.columns))
    print(f"data column---{write_fields}")
    final_res.write.format("doris") \
        .option("doris.table.identifier",
                f"{properties_config['doris_db']}.{properties_config['doris_wat_results_table']}") \
        .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
        .option("user", f"{properties_config['doris_user']}") \
        .option("password", f"{properties_config['doris_password']}") \
        .option("doris.write.fields", write_fields) \
        .option("doris.sink.batch.interval.ms", 50) \
        .option("doris.sink.batch.size", 10000) \
        .save()
    time4 = datetime.now()
    print(f"算法结果一共有{final_res.count()}条")
    print("算法结果写回数据库消耗的时间是：", time4 - time3)
    final_res.unpersist()


def process_record_by_site(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_parametric_name, good_site, bad_site = parse_JSON_config(
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

    query_sql = build_wat_query(parse_dict, properties_config)
    print(query_sql)

    time1 = datetime.now()
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    time2 = datetime.now()
    print("从数据库中获取数据消耗的时间是：", time2 - time1)

    doris_spark_df.persist(StorageLevel.MEMORY_AND_DISK)
    final_res = ExertWatBySite.fit_by_site_model(df=doris_spark_df,
                                                 request_id=request_id,
                                                 grpby_list=grpby_list,
                                                 merge_operno_list=merge_operno,
                                                 merge_prodg1_list=merge_prodg1,
                                                 merge_product_list=merge_product,
                                                 merge_parametric_name_list=merge_parametric_name,
                                                 good_site_columns=good_site,
                                                 bad_site_columns=bad_site)
    time3 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time3 - time2)

    # 需要写的列
    write_fields = ','.join(map(str, final_res.columns))
    print(f"data column---{write_fields}")
    final_res.write.format("doris") \
        .option("doris.table.identifier",
                f"{properties_config['doris_db']}.{properties_config['doris_wat_results_table']}") \
        .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
        .option("user", f"{properties_config['doris_user']}") \
        .option("password", f"{properties_config['doris_password']}") \
        .option("doris.write.fields", write_fields) \
        .option("doris.sink.batch.interval.ms", 50) \
        .option("doris.sink.batch.size", 10000) \
        .save()
    time4 = datetime.now()
    print(f"算法结果一共有{final_res.count()}条")
    print("算法结果写回数据库消耗的时间是：", time4 - time3)
    final_res.unpersist()


def process_record_by_zone(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_parametric_name, good_site, bad_site = parse_JSON_config(
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

    query_sql = build_wat_query(parse_dict, properties_config)
    print(query_sql)

    time1 = datetime.now()
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    time2 = datetime.now()
    print("从数据库中获取数据消耗的时间是：", time2 - time1)

    request_params = df_info_["requestParam"].values[0]
    parse_dict = json.loads(request_params)
    upload_id = parse_dict.get("uploadId")
    db_config = {
        'host': properties_config['doris_ip'],
        'port': 9030,
        'user': properties_config['doris_user'],
        'password': properties_config['doris_password'],
        'database': properties_config['doris_db']
    }
    query = f"SELECT NAME, MODE, SITES, ANALYSIS, LABEL FROM rca.CONF_MODE WHERE ANALYSIS = 'WAT' AND UPLOAD_ID='{upload_id}'"

    by_zone_config_pare = ByZoneConfigPare(**db_config)
    mode_info_json = by_zone_config_pare.run(query)

    final_res = ExertWatByZone.fit_by_zone_model(df=doris_spark_df,
                                                 request_id=request_id,
                                                 grpby_list=grpby_list,
                                                 merge_operno_list=merge_operno,
                                                 merge_prodg1_list=merge_prodg1,
                                                 merge_product_list=merge_product,
                                                 merge_parametric_name_list=merge_parametric_name,
                                                 mode_info=mode_info_json.get("mode_info"))
    time3 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time3 - time2)

    # 需要写的列
    write_fields = ','.join(map(str, final_res.columns))
    print(f"data column---{write_fields}")
    final_res.write.format("doris") \
        .option("doris.table.identifier",
                f"{properties_config['doris_db']}.{properties_config['doris_wat_results_table']}") \
        .option("doris.fenodes", f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}") \
        .option("user", f"{properties_config['doris_user']}") \
        .option("password", f"{properties_config['doris_password']}") \
        .option("doris.write.fields", write_fields) \
        .option("doris.sink.batch.interval.ms", 50) \
        .option("doris.sink.batch.size", 10000) \
        .save()
    time4 = datetime.now()
    print(f"算法结果一共有{final_res.count()}条")
    print("算法结果写回数据库消耗的时间是：", time4 - time3)


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

    try:
        # PARAMETRICNAME的部分合并结果
        merge_parametric_name = list(parse_dict.get('mergedParametricName')) if parse_dict.get('mergedParametricName') else None
    except KeyError:
        merge_parametric_name = None

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
        group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO"]
        # PRODUCT_ID, PROG1, CHAMBER 这3个存在一键合并的切换开关
        # 且一键合并PROG1时会自动一键合并PRODUCT_ID
        flag_merge_prodg1 = parse_dict.get('flagMergeAllProdg1')
        flag_merge_product_id = parse_dict.get('flagMergeAllProductId')
        flag_merge_chamber = parse_dict.get('flagMergeAllChamber')

        if flag_merge_prodg1 == '1':
            # 一键合并PROG1时，部分合并PROG1和PRODUCT_ID的情况都会被忽略
            merge_prodg1 = None
            merge_product = None
            group_by_list = ["OPE_NO"]
        elif flag_merge_product_id == '1':
            # 一键合并PRODUCT_ID时，部分合并PRODUCT_ID的情况会被忽略
            merge_product = None
            group_by_list = ["PRODG1","OPE_NO"]

    return parse_dict, request_id, group_by_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_parametric_name, good_site, bad_site
