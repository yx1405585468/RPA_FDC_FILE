import json
from typing import Optional

keyword_map_from_json_to_table: dict = {
    "prodg1": "PRODG1",
    "waferId": "WAFER_ID",
    "dateRange": "START_TIME",
    "productId": "PRODUCT_ID",
    "operNo": "OPE_NO",
    "eqp": "EQP_NAME",
    "tool": "CHAMBER_NAME",
    "chamber": "CHAMBER_NAME",
    "lot": "LOT_ID",
    "recipeName": "RECIPE_NAME"
}


def process_like(key: str, value: list[str]) -> str:
    # 处理模糊条件的匹配: (key like 'aa%' or key like "bb%")
    key = keyword_map_from_json_to_table.get(key)
    v_join = ' or '.join([f"d1.{key} like  '{v.replace('*', '%')}' " for v in value])
    return "({})".format(v_join)


def process_not_like(key: str, value: list[str]) -> str:
    # 处理非模糊条件的匹配:key in ('aa', 'bb')
    key = keyword_map_from_json_to_table.get(key)
    v_join = ",".join(["'{0}'".format(v.replace("\\", "\\\\")) for v in value])
    return "d1.{} in ({})".format(key, v_join)


def concat_time_filter_sql_with_other_keyword_sql(time_filter_sql: str, other_keyword_sql: str) -> str:
    """
    拼接时间过滤条件与非时间过滤条件
    :param time_filter_sql:
    :param other_keyword_sql:
    :return:
    """
    time_strip = time_filter_sql.strip()
    other_strip = other_keyword_sql.strip()
    if len(time_strip) == 0 and len(other_strip) == 0:
        return ""
    elif len(time_strip) != 0 and len(other_strip) == 0:
        return time_filter_sql
    elif len(time_strip) == 0 and len(other_strip) != 0:
        return other_keyword_sql
    else:
        return f'{time_filter_sql} and {other_keyword_sql}'


def get_time_selection_sql(time_keyword, max_time=None, min_time=None):
    """
    获取时间区间的筛选的sql, 起始时间和结束时间都是可选的
    :param time_keyword:
    :param max_time:
    :param min_time:
    :return:
    """
    # 根据取值，生成单个时间过滤条件
    if min_time:
        time_part_min = f"d1.{time_keyword} >= '{min_time}'"
    else:
        time_part_min = " "

    if max_time:
        time_part_max = f"d1.{time_keyword} < '{max_time}'"
    else:
        time_part_max = " "

    # 如果存在，拼接多个查询条件，或者只保留一个过滤条件
    if (max_time is not None) and (min_time is not None):
        time_sql = f' {time_part_min} and {time_part_max}'
    elif (max_time is None) and (min_time is None):
        time_sql = " "
    else:
        time_sql = time_part_max if max_time else time_part_min

    return time_sql


def process_one_keyword(key, value: list[str]) -> Optional[str]:
    if len(value) == 0:
        return None

    not_like_list = [v for v in value if "*" not in v]
    like_list = [v for v in value if "*" in v]

    # 处理模糊条件
    if len(not_like_list) != 0:
        not_like_sql_str = process_not_like(key, not_like_list)
    else:
        not_like_sql_str = ""

    # 处理非模糊条件

    if len(like_list) != 0:
        like_sql_str = process_like(key, like_list)
    else:
        like_sql_str = ""

    # 去除为一个元素为空字符串的情况的情况的情况
    concat_sql_str_list = [sql_str for sql_str in [like_sql_str, not_like_sql_str] if len(sql_str) != 0]
    # 使用or 操作 单字段过滤 的like 和 not like 语句
    return "(" + " or ".join(concat_sql_str_list) + ")"


def build_uva_query(select_condition_dict: dict, properties_config: dict) -> str:
    fd_uva_table = f"{properties_config['doris_db']}.{properties_config['doris_fd_uva_table']}"
    uploaded_wafer_table_name = f"{properties_config['doris_db']}.{properties_config['doris_uploaded_wafer_table']}"

    # 查询条件转sql,并打标签，label '0': good wafer, '1': bad wafer
    filter_sql_list = []
    for keyword, value in select_condition_dict.items():
        if keyword in ["prodg1", "productId", "eqp", "tool", "chamber", "recipeName", "operNo"]:
            sql_filter_one_keyword = process_one_keyword(keyword, value)
            if sql_filter_one_keyword is not None:
                filter_sql_list.append(sql_filter_one_keyword)

    # 处理时间区间
    time_bin = select_condition_dict.get("dateRange")

    if len(time_bin) >= 1:  # list[dict]
        min_time = time_bin.get("start")
        max_time = time_bin.get("end")
    else:
        min_time = None
        max_time = None

    # 处理waferId
    # waferId = select_condition_dict.get("waferId")
    # good_wafer_list = waferId.get("good")
    # bad_wafer_list = waferId.get("bad")
    upload_id = select_condition_dict.get("uploadId")

    # 根据time 过滤条件 生成sql
    time_filter_sql = get_time_selection_sql(time_keyword=keyword_map_from_json_to_table.get('dateRange'),
                                             max_time=max_time, min_time=min_time)

    # if len(good_wafer_list) > 0 and len(bad_wafer_list) > 0:
    if upload_id is not None and len(upload_id) > 0:

        other_keyword_filter = " and ".join(filter_sql_list)

        case_when_statment = f"""  (case
        when d2.label = 'GOOD' then 0 
        else 1
        end ) as label
        """

        filter_sql_concat = concat_time_filter_sql_with_other_keyword_sql(time_filter_sql, other_keyword_filter)

        if filter_sql_concat != '':
            select_sql = f"""SELECT *, {case_when_statment} from {fd_uva_table} d1  
            join {uploaded_wafer_table_name} d2
        on d1.WAFER_ID = d2.name
            where {filter_sql_concat} and d2.UPLOAD_ID = '{upload_id}'"""
        else:
            select_sql = f"""SELECT *, {case_when_statment} from {fd_uva_table} d1  
            join {uploaded_wafer_table_name} d2
        on d1.WAFER_ID = d2.name where d2.UPLOAD_ID = '{upload_id}'"""

    else:
        raise ValueError("good bad wafer 都必须选！")

    # print(select_sql)
    select_sql = select_sql.replace("*",
                                    "d1.WAFER_ID, d1.CHAMBER_ID, d1.RUN_ID, d1.EQP_NAME, d1.PRODUCT_ID, d1.PRODG1, d1.CHAMBER_NAME, d1.LOT_ID, d1.RECIPE_NAME, d1.OPE_NO, d1.PARAMETRIC_NAME, d1.CASE_INFO, d1.STATUS, d1.RESULT")

    if fd_uva_table == f"{properties_config['doris_db']}.{properties_config['doris_fd_uva_table']}":
        select_sql = f"{select_sql} and d1.STATUS != 'ERROR'"

    print("select_sql", select_sql)
    return select_sql

if __name__ == '__main__':
    request_params = '{"dateRange":{"start":"2021-12-06 10:51:40","end":"2024-03-06 10:51:40"},"operNo":["1U.CDG10","1U.CDG20","1V.PQA10","2U.PQA10","2V.PQW10","3U.PQA10","6V.CDG10","7U.PQA10","7U.PQX10","TM.PQX10","XX.PQW01","XX.PQX02","1U.EQW20","1U.PQW10","1U.PQX10","1V.PQX10","1V.PQX20","2U.PQW10","2U.PQX10","2V.EQW10","3U.PQX10","6V.EQW10","6V.PQA10","7U.PQW10","PV.PQX10","TM.EQW10","TV.PQA10","XX.CCX01","XX.CCY01","1C.PQA10","1C.PQX10","1V.PQW10"],"uploadId":"a1291cd1d662438093cfd2616a4b72ad","goodSite":["SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE8_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE6_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE7_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE1_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE4_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE2_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE3_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL","SITE5_VAL"],"badSite":["SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE16_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE27_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE19_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE47_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE31_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE37_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE45_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE11_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE14_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE17_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE20_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE22_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE25_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE38_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE39_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE48_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE9_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE10_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE28_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE29_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE36_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE43_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE44_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE46_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE18_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE33_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE35_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE40_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE30_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE32_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE34_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE41_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE42_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE15_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE21_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE24_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE12_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE13_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE23_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE26_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE49_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL","SITE50_VAL"],"flagMergeAllProdg1":"0","flagMergeAllProductId":"0","flagMergeAllChamber":"0","mergeProdg1":[],"mergeProductId":[],"mergeEqp":[],"mergeChamber":[],"mergeOperno":[{"1U.CDG10,1U.CDG20,1V.PQA10,2U.PQA10,2V.PQW10,3U.PQA10,6V.CDG10,7U.PQA10,7U.PQX10,TM.PQX10,XX.PQW01,XX.PQX02,1U.EQW20,1U.PQW10,1U.PQX10,1V.PQX10,1V.PQX20,2U.PQW10,2U.PQX10,2V.EQW10,3U.PQX10,6V.EQW10,6V.PQA10,7U.PQW10,PV.PQX10,TM.EQW10,TV.PQA10,XX.CCX01,XX.CCY01,1C.PQA10,1C.PQX10":["1U.CDG10","1U.CDG20","1V.PQA10","2U.PQA10","2V.PQW10","3U.PQA10","6V.CDG10","7U.PQA10","7U.PQX10","TM.PQX10","XX.PQW01","XX.PQX02","1U.EQW20","1U.PQW10","1U.PQX10","1V.PQX10","1V.PQX20","2U.PQW10","2U.PQX10","2V.EQW10","3U.PQX10","6V.EQW10","6V.PQA10","7U.PQW10","PV.PQX10","TM.EQW10","TV.PQA10","XX.CCX01","XX.CCY01","1C.PQA10","1C.PQX10"]}]}'
    parse_dict = json.loads(request_params)
    properties_config = dict(doris_db='rca', doris_fd_uva_table='rca_uva_table', doris_uploaded_wafer_table='rca_uploaded_wafer_table')
    print(build_uva_query(parse_dict, properties_config))
