
import pyspark.pandas as ps
from pyspark.sql.functions import pandas_udf, PandasUDFType, countDistinct, when, col, rank, lit



#









if __name__ == "__main__":
    import warnings
    import pandas as pd
    import os
    warnings.filterwarnings('ignore')

    # local模式
    # import findspark
    # findspark.init()
    # spark = SparkSession \
    #     .builder \
    #     .appName("ywj") \
    #     .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
    #     .master("local[*]") \
    #     .getOrCreate()

    # spark集群模式
    from pyspark.sql import SparkSession

    os.environ['PYSPARK_PYTHON'] = '/usr/local/python-3.9.13/bin/python3'
    spark = SparkSession.builder \
        .appName("pandas_udf") \
        .config('spark.sql.session.timeZone', 'Asia/Shanghai') \
        .config("spark.scheduler.mode", "FAIR") \
        .config('spark.driver.memory', '1024m') \
        .config('spark.driver.cores', '3') \
        .config('spark.executor.memory', '1024m') \
        .config('spark.executor.cores', '1') \
        .config('spark.cores.max', '2') \
        .config('spark.driver.host', '192.168.22.28') \
        .master("spark://192.168.12.47:7077,192.168.12.48:7077") \
        .getOrCreate()

    # 读取数据
    df_run_ = pd.read_csv('C:/Users/yang.wenjun/Desktop/晶合FDC-freedemo资料/df_defect.csv')
    df_run_spark = ps.from_pandas(df_run_).to_spark()
    print(df_run_spark.count())
    df_run_spark.show()

    grps = (df_run_spark.groupBy(['OPER_NO', 'RECIPE_ID']).agg(countDistinct('WAFER_ID').alias('wafer_count'),
                                                                   countDistinct('WAFER_ID', when(df_run_spark['label'] == 0, 1)).alias('good_num'),
                                                                   countDistinct('WAFER_ID', when(df_run_spark['label'] == 1, 1)).alias('bad_num')))
    grps.show()




