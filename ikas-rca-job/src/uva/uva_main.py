import json
import pandas as pd
from datetime import datetime
from src.utils import read_jdbc_executor
from src.utils.score_executor import get_score_df
from src.uva.build_query import build_uva_query
from src.uva.uva_algorithm import ExertUvaAlgorithm


def process_record(sparkSession, json_config, properties_config):
    df_info_ = pd.DataFrame(
        {
            "requestId": [json_config["requestId"]],
            "requestParam": [json.dumps(json_config["requestParam"])],
        }
    )

    # 解析JSON并且读取数据
    (
        parse_dict,
        request_id,
        grpby_list,
        merge_operno,
        merge_prodg1,
        merge_product,
        merge_eqp,
        merge_chamber,
    ) = parse_JSON_config(df_info_)
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
    print(query_sql)

    time1 = datetime.now()
    doris_spark_df = read_jdbc_executor.read(sparkSession, query_sql, properties_config)
    doris_spark_df.persist()
    time2 = datetime.now()
    print("从数据库中获取数据消耗的时间是：", time2 - time1)

    final_res = ExertUvaAlgorithm.fit_uva_model(
        df=doris_spark_df,
        grpby_list=grpby_list,
        request_id=request_id,
        merge_operno_list=merge_operno,
        merge_prodg1_list=merge_prodg1,
        merge_product_list=merge_product,
        merge_eqp_list=merge_eqp,
        merge_chamber_list=merge_chamber,
    )
    time3 = datetime.now()
    print("算法运行得到最终结果消耗的时间是：", time3 - time2)

    # 需要写的列
    # write_fields = ",".join(map(str, final_res.columns))
    # print(f"data---{write_fields}")
    score_df = get_score_df(spark_session=sparkSession, properties_config=properties_config, data_frame=final_res,
                            analysis_type="FDC")
    score_df.write.format("doris").option(
        "doris.table.identifier",
        f"{properties_config['doris_db']}.{properties_config['doris_uva_results_table']}",
    ).option(
        "doris.fenodes",
        f"{properties_config['doris_ip']}:{properties_config['doris_fe_http_port']}",
    ).option(
        "user", f"{properties_config['doris_user']}"
    ).option(
        "password", f"{properties_config['doris_password']}"
    ).option(
        "doris.sink.batch.size", 10000
    ).option(
        "doris.sink.max-retries", 3
    ).option(
        "doris.sink.auto-redirect", True
    ).save()
    # score_df.show(10)
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
        merge_operno = (
            list(parse_dict.get("mergeOperno"))
            if parse_dict.get("mergeOperno")
            else None
        )
    except KeyError:
        merge_operno = None

    try:
        # PROG1的部分合并结果
        merge_prodg1 = (
            list(parse_dict.get("mergeProdg1"))
            if parse_dict.get("mergeProdg1")
            else None
        )
    except KeyError:
        merge_prodg1 = None

    try:
        # PRODUCT_ID的部分合并结果
        merge_product = (
            list(parse_dict.get("mergeProductId"))
            if parse_dict.get("mergeProductId")
            else None
        )
    except KeyError:
        merge_product = None

    try:
        # EQP的部分合并结果
        merge_eqp = (
            list(parse_dict.get("mergeEqp")) if parse_dict.get("mergeEqp") else None
        )
    except KeyError:
        merge_eqp = None

    try:
        # CHAMBER的部分合并结果
        merge_chamber = (
            list(parse_dict.get("mergeChamber"))
            if parse_dict.get("mergeChamber")
            else None
        )
    except KeyError:
        merge_chamber = None

    # group by 子句中的字段
    group_by_list = parse_dict.get("groupByList")
    if group_by_list is None or len(group_by_list) == 0:
        group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO", "EQP_NAME", "CHAMBER_NAME"]
        # PRODUCT_ID, PROG1, CHAMBER 这3个存在一键合并的切换开关
        # 且一键合并PROG1时会自动一键合并PRODUCT_ID
        flag_merge_prodg1 = parse_dict.get("flagMergeAllProdg1")
        flag_merge_product_id = parse_dict.get("flagMergeAllProductId")
        flag_merge_chamber = parse_dict.get("flagMergeAllChamber")

        if flag_merge_prodg1 == "1":
            # 一键合并PROG1时，部分合并PROG1和PRODUCT_ID的情况都会被忽略
            merge_prodg1 = None
            merge_product = None
            group_by_list = ["OPE_NO", "EQP_NAME", "CHAMBER_NAME"]
            if flag_merge_chamber == "1":
                group_by_list = ["OPE_NO", "EQP_NAME"]
        elif flag_merge_product_id == "1":
            # 一键合并PRODUCT_ID时，部分合并PRODUCT_ID的情况会被忽略
            merge_product = None
            group_by_list = ["PRODG1", "OPE_NO", "EQP_NAME", "CHAMBER_NAME"]
            if flag_merge_chamber == "1":
                # 一键合并CHAMBER时，部分合并CHAMBER的情况会被忽略
                group_by_list = ["PRODG1", "OPE_NO", "EQP_NAME"]
        elif flag_merge_chamber == "1":
            merge_chamber = None
            group_by_list = ["PRODG1", "PRODUCT_ID", "OPE_NO", "EQP_NAME"]

    return (
        parse_dict,
        request_id,
        group_by_list,
        merge_operno,
        merge_prodg1,
        merge_product,
        merge_eqp,
        merge_chamber,
    )
