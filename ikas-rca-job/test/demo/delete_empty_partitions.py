# -*- coding: utf-8 -*-
# time: 2024/7/18 21:40
# file: delete_empty_partitions.py
# author: way lee
from datetime import datetime

import pymysql
from pymysql import cursors

# 数据库连接配置
DB_CONFIG = {
    'host': '192.168.13.17',
    'port': 9030,
    'user': 'root',
    'password': '123456',
    'db': 'test',
    'charset': 'utf8mb4',
    'cursorclass': cursors.DictCursor
}

TABLE_NAME = 'DWD_FD_UVA_DATA_TEST'
PARTITIONS_FILE = 'partitions.txt'
LOG_FILE = f'{TABLE_NAME}_partition_cleanup.log'

# ANSI escape codes for colors
GREEN = '\033[1;92m'
YELLOW = '\033[1;93m'
RED = '\033[1;91m'
ENDC = '\033[0m'


def log_message(message, color=None):
    if color:
        print(f"{color}{message}{ENDC}")
    else:
        print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')


def main():
    # 初始化日志文件
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("Doris 分区清理日志\n")
        log_file.write(f"Date: {datetime.now()}\n")
        log_file.write("---------------------------------\n")
    log_message(f"正在处理表: {TABLE_NAME}", YELLOW)
    # 连接到数据库
    connection = pymysql.connect(**DB_CONFIG)

    try:
        with connection.cursor() as cursor:
            # 关闭动态分区
            cursor.execute(f'ALTER TABLE {TABLE_NAME} SET ("dynamic_partition.enable" = "false");')
            connection.commit()
            log_message("已关闭动态分区", GREEN)

            # 读取分区文件并遍历每个分区
            with open(PARTITIONS_FILE, 'r') as partitions_file:
                for partition in partitions_file:
                    partition = partition.strip()
                    cursor.execute(f'SELECT COUNT(1) FROM {TABLE_NAME} PARTITION ({partition});')
                    count = cursor.fetchone()['count(1)']

                    # 如果分区没有数据，删除该分区并记录日志
                    if count == 0:
                        log_message(f"发现空分区: {TABLE_NAME} 分区 {partition}", YELLOW)
                        cursor.execute(f'ALTER TABLE {TABLE_NAME} DROP PARTITION {partition};')
                        connection.commit()
                        log_message(f"已删除分区: {partition} from 表: {TABLE_NAME}", RED)

            # 删除完成后开启动态分区
            cursor.execute(f'ALTER TABLE {TABLE_NAME} SET ("dynamic_partition.enable" = "true");')
            connection.commit()
            log_message("已重新开启动态分区", GREEN)

    finally:
        connection.close()

    log_message(f"完成处理表: {TABLE_NAME}", YELLOW)
    log_message("---------------------------------")
    log_message("所有表的空分区清理已完成。")


if __name__ == "__main__":
    main()
