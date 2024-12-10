import json
import pandas as pd
from datetime import datetime
from src.utils import read_jdbc_executor
from src.defect.defect_algorithm import ExertDefectAlgorithm

# defect算法 根据实际情况改写下面的表名和导入代码  from src.defect.build_query import build_defect_query
from src.defect.build_query import build_defect_query


def process_record_defect(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame({"requestId": [json_config["requestId"]],
                             "requestParam": [json.dumps(json_config["requestParam"])]})

    parse_dict, request_id, grpby_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_layerid, good_site, bad_site = parse_JSON_config(
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
    print("merge_layerid:")
    print(merge_layerid)

    query_sql = build_defect_query(parse_dict, properties_config)
    print(query_sql)

    time1 = datetime.now()
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    doris_spark_df.persist()
    time2 = datetime.now()
    print("从数据库中获取数据消耗的时间是：", time2 - time1)

    final_res = ExertDefectAlgorithm.fit_defect_model(df=doris_spark_df,
                                                      request_id=request_id,
                                                      merge_layerid_list=merge_layerid,
                                                      merge_prodg1_list=merge_prodg1,
                                                      merge_product_list=merge_product,
                                                      grpby_list=grpby_list)
    time3 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time3 - time2)

    # 需要写的列
    write_fields = ','.join(map(str, final_res.columns))
    print(f"data column---{write_fields}")
    final_res.write.format("doris") \
        .option("doris.table.identifier",
                f"{properties_config['doris_db']}.{properties_config['doris_defect_results_table']}") \
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
        # LAYER_ID的部分合并结果
        merge_layerid = list(parse_dict.get('mergeLayerId')) if parse_dict.get('mergeLayerId') else None
    except KeyError:
        merge_layerid = None

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
        group_by_list = ["PRODG1", "PRODUCT_ID", "LAYER_ID"]
        # PRODUCT_ID, PROG1, CHAMBER 这3个存在一键合并的切换开关
        # 且一键合并PROG1时会自动一键合并PRODUCT_ID
        flag_merge_prodg1 = parse_dict.get('flagMergeAllProdg1')
        flag_merge_product_id = parse_dict.get('flagMergeAllProductId')
        flag_merge_chamber = parse_dict.get('flagMergeAllChamber')

        if flag_merge_prodg1 == '1':
            # 一键合并PROG1时，部分合并PROG1和PRODUCT_ID的情况都会被忽略
            merge_prodg1 = None
            merge_product = None
            group_by_list = ['LAYER_ID']
        elif flag_merge_product_id == '1':
            # 一键合并PRODUCT_ID时，部分合并PRODUCT_ID的情况会被忽略
            merge_product = None
            group_by_list = ["PRODG1", "LAYER_ID"]

    return parse_dict, request_id, group_by_list, merge_operno, merge_prodg1, merge_product, merge_eqp, merge_chamber, merge_layerid, good_site, bad_site
