import re

from src.utils.config_utils import read_properties_file


def rca_test():
    # file = read_properties_file("D:\ideaproject\ikas-rca-job\tests\app_config.properties")

    config_lines = [
        'rca_http_api_url = http://192.168.13.17:8089/internal/task/end',
        '# doris',
        'doris_ip = 192.168.13.17',
        'doris_fe_http_port = 18030',
        'doris_jdbc_url = jdbc:mysql://192.168.13.17:9030/rca',
        'doris_user = root',
        'doris_password = 123456',
        '# doris table',
        'doris_db = rca',
        'doris_fd_uva_table = DWD_FD_UVA_DATA',
        'doris_inline_wafer_summary_table = DWD_INLINE_WAFER_SUMMARY',
        'doris_uploaded_wafer_table = conf_wafer',
        'doris_uva_results_table = uva_results',
        'doris_inline_results_table = inline_results'
    ]
    # 正则表达式匹配键值对，允许=号前后有空格
    pattern = re.compile(r'\s*(.*?)\s*=\s*(.*)')
    properties = {}
    for line in config_lines:
        # 跳过空行和注释行
        if line.strip() == "" or line.strip().startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            key, value = match.groups()
            properties[key] = value
    print("load_config_properties:")
    print(properties)


if __name__ == '__main__':
    rca_test()
