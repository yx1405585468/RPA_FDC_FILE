	DEFECT
		过程：
			-- 1 获取wafer_id
			SELECT
				CONCAT(SUBSTR(FOUP_ID, 1, 6), '-', CASE WHEN FOUPSLOT_ID < 10 THEN CONCAT('0', FOUPSLOT_ID) WHEN FOUPSLOT_ID >= 10 THEN FOUPSLOT_ID END) AS WAFER_ID
			FROM
			(
			SELECT distinct a.FOUP_ID,a.FOUPSLOT_ID,a.EQUIPMENT_ID,a.MEASURE_TIME AS CLAIM_TIME,a.ed_parameter_id AS DCITEM_NAME,a.VALUE AS DCITEM_VALUE,a.CONTROL_HIGH,a.CONTROL_LOW,a.SPEC_HIGH,a.SPEC_LOW,a.OOC_RESULT,a.OOS_RESULT,a.PROCESS_EQUIP AS PROC_EQP_ID,a.PROCESS_JOB_ID,a.QC_JOB_ID,a.PURPOSE,b.JOB_SPEC_GROUP_ID,a.RECIPE,a.MEASURE_TIME AS DC_Time
			FROM DWD_NEDA.DWD_OCS_WAFER_SUMMARY a,
			(select FOUP_ID ,PTOOL_ID ,CHAMBERS ,JOB_SPEC_GROUP_ID , JOB_SPEC_GROUP_PO_ID FROM ODS_OCS_PROD_A1A2.ODS_V_JOB_SPEC_GROUP_H m
			union all
			select FOUP_ID ,PTOOL_ID ,CHAMBERS ,JOB_SPEC_GROUP_ID , JOB_SPEC_GROUP_PO_ID FROM ODS_OCS_PROD_A3.ODS_V_JOB_SPEC_GROUP_H n ) b
			where b.JOB_SPEC_GROUP_PO_ID =a.JOB_GROUP_ID
			AND a.PROCESS_EQUIP = 'CMA51'
			AND a.MEASURE_TIME >= DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND a.MEASURE_TIME < DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR) -- 时间段，客户时间前后各推3个小时
			) tmp
			GROUP BY FOUP_ID, FOUPSLOT_ID;

			-- 2 根据第一步返回的wafer_id，获取wafer_key
			SELECT
				WAFER_KEY
			FROM
			(
			SELECT
			    case when WAFER_ID regexp '^[@#][[:digit:]]{2}$' then concat(SUBSTR(LOT_ID, 1, 6), '-', SUBSTR(WAFER_ID, 2, 2))
			         when WAFER_ID regexp '^[[:digit:]]{2}$' then concat(SUBSTR(LOT_ID, 1, 6), '-', SUBSTR(WAFER_ID, 1, 2))
			         when length(WAFER_ID)=9 and SUBSTR(LOT_ID, 1, 6)=SUBSTR(WAFER_ID, 1, 6) and SUBSTR(WAFER_ID, 7, 1)='-' then WAFER_ID
			    end WAFER_ID,
			    WAFER_KEY ,
			    INSPECTION_TIME
			FROM
			    ODS_UDB.ODS_INSP_WAFER_SUMMARY
			) TMP
			WHERE WAFER_ID IN 
			('ATN019-02','ATN019-11','ATN039-08','ATN016-07','ATN019-04','ATN016-02','ATN019-08','ATN019-06','ATP074-13','ATN019-12','ATP068-24','ATP068-22','ATN016-01',
			'ATP074-12','ATN019-07','ATN019-01','ATP068-25','ATP068-23','ATN019-10','ATP074-22','ATN019-09','ATN019-05','ATN019-03','ATP074-23','ATN016-09','ATN016-05') -- 第一步返回的wafer_id
			AND INSPECTION_TIME >= DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND INSPECTION_TIME < DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR)
			ORDER BY INSPECTION_TIME
			;

			-- 3 根据第二步返回的wafer_key，获取defect数据
			SELECT
				((isd.wafer_x/1000)-(iws.center_x/1000)) / 1000 - (CENTER_X-ORIGIN_X) / 1000000 as x,
				((isd.wafer_y/1000)-(iws.center_y/1000))  / 1000 - (CENTER_Y-ORIGIN_Y) / 1000000 as y,
				SIZE_D/1000 AS DIAMETER,
				isd.WAFER_KEY,
				isd.INSPECTION_TIME
			FROM
				ODS_UDB.ODS_INSP_DEFECT isd,
				ODS_UDB.ODS_INSP_RECIPE ire,
				ODS_UDB.ODS_INSP_WAFER_SUMMARY iws
			WHERE
				isd.INSPECTION_TIME = iws.INSPECTION_TIME
				AND ire.RECIPE_KEY = iws.RECIPE_KEY
				AND isd.WAFER_KEY IN (6992681,6993889,6993910,6993931,6993949,6994750,6994753,6994755,6994760)
				AND isd.INSPECTION_TIME BETWEEN DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR);


			-- 4 在 3 的结果上，根据 wafer_key，去 ODS_UDB.ODS_INSP_WAFER_SUMMARY
			-- 通过 LOT_ID 取LOT_ID(LOT_ID 的 前8个字符) 和 WAFER_ID(LOT_ID前6个字符 + "-" + 两个-中间的部分)
			-- 举例 表中的LOT_ID=ATP06821-25-04, 则lot_id=ATP06821，wafer_id=ATP068-25
			SELECT 
				TMP1.x,
				TMP1.Y,
				TMP1.DIAMETER,
				TMP1.WAFER_KEY,
				TMP1.INSPECTION_TIME,
				SUBSTR(TMP2.LOT_ID, 1, 6) AS LOT_ID,
				CONCAT(SUBSTR(TMP2.LOT_ID, 1, 6),SUBSTR(SUBSTR(TMP2.LOT_ID, 9), 1, 3)) AS WAFER_ID
			FROM
			(
			SELECT
				((isd.wafer_x/1000)-(iws.center_x/1000)) / 1000 - (CENTER_X-ORIGIN_X) / 1000000 as x,
				((isd.wafer_y/1000)-(iws.center_y/1000))  / 1000 - (CENTER_Y-ORIGIN_Y) / 1000000 as y,
				SIZE_D/1000 AS DIAMETER,
				isd.WAFER_KEY AS WAFER_KEY,
				isd.INSPECTION_TIME AS INSPECTION_TIME
			FROM
				ODS_UDB.ODS_INSP_DEFECT isd,
				ODS_UDB.ODS_INSP_RECIPE ire,
				ODS_UDB.ODS_INSP_WAFER_SUMMARY iws
			WHERE
				isd.INSPECTION_TIME = iws.INSPECTION_TIME
				AND ire.RECIPE_KEY = iws.RECIPE_KEY
				AND isd.WAFER_KEY IN (6992681,6993889,6993910,6993931,6993949,6994750,6994753,6994755,6994760)
				AND isd.INSPECTION_TIME BETWEEN DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR)
			) TMP1
			LEFT JOIN
			ODS_UDB.ODS_INSP_WAFER_SUMMARY TMP2
			ON TMP1.WAFER_KEY = TMP2.WAFER_KEY;

		-- 合成SQL
			SELECT 
				TMP1.x,
				TMP1.Y,
				TMP1.DIAMETER,
				TMP1.WAFER_KEY,
				TMP1.INSPECTION_TIME,
				SUBSTR(TMP2.LOT_ID, 1, 6) AS LOT_ID,
				CONCAT(SUBSTR(TMP2.LOT_ID, 1, 6),SUBSTR(SUBSTR(TMP2.LOT_ID, 9), 1, 3)) AS WAFER_ID
			FROM
			(
			SELECT
				((isd.wafer_x/1000)-(iws.center_x/1000)) / 1000 - (CENTER_X-ORIGIN_X) / 1000000 as x,
				((isd.wafer_y/1000)-(iws.center_y/1000))  / 1000 - (CENTER_Y-ORIGIN_Y) / 1000000 as y,
				SIZE_D/1000 AS DIAMETER,
				isd.WAFER_KEY AS WAFER_KEY,
				isd.INSPECTION_TIME AS INSPECTION_TIME
			FROM
				ODS_UDB.ODS_INSP_DEFECT isd,
				ODS_UDB.ODS_INSP_RECIPE ire,
				ODS_UDB.ODS_INSP_WAFER_SUMMARY iws
			WHERE
				isd.INSPECTION_TIME = iws.INSPECTION_TIME
				AND ire.RECIPE_KEY = iws.RECIPE_KEY
				AND isd.WAFER_KEY IN 
					(
						SELECT
							WAFER_KEY
						FROM
						(
						SELECT
						    case when WAFER_ID regexp '^[@#][[:digit:]]{2}$' then concat(SUBSTR(LOT_ID, 1, 6), '-', SUBSTR(WAFER_ID, 2, 2))
						         when WAFER_ID regexp '^[[:digit:]]{2}$' then concat(SUBSTR(LOT_ID, 1, 6), '-', SUBSTR(WAFER_ID, 1, 2))
						         when length(WAFER_ID)=9 and SUBSTR(LOT_ID, 1, 6)=SUBSTR(WAFER_ID, 1, 6) and SUBSTR(WAFER_ID, 7, 1)='-' then WAFER_ID
						    end WAFER_ID,
						    WAFER_KEY ,
						    INSPECTION_TIME
						FROM
						    ODS_UDB.ODS_INSP_WAFER_SUMMARY
						) TMP
						WHERE WAFER_ID IN 
							(
								SELECT
									CONCAT(SUBSTR(FOUP_ID, 1, 6), '-', CASE WHEN FOUPSLOT_ID < 10 THEN CONCAT('0', FOUPSLOT_ID) WHEN FOUPSLOT_ID >= 10 THEN FOUPSLOT_ID END) AS WAFER_ID
								FROM
								(
								SELECT distinct a.FOUP_ID,a.FOUPSLOT_ID,a.EQUIPMENT_ID,a.MEASURE_TIME AS CLAIM_TIME,a.ed_parameter_id AS DCITEM_NAME,a.VALUE AS DCITEM_VALUE,a.CONTROL_HIGH,a.CONTROL_LOW,a.SPEC_HIGH,a.SPEC_LOW,a.OOC_RESULT,a.OOS_RESULT,a.PROCESS_EQUIP AS PROC_EQP_ID,a.PROCESS_JOB_ID,a.QC_JOB_ID,a.PURPOSE,b.JOB_SPEC_GROUP_ID,a.RECIPE,a.MEASURE_TIME AS DC_Time
								FROM DWD_NEDA.DWD_OCS_WAFER_SUMMARY a,
								(select FOUP_ID ,PTOOL_ID ,CHAMBERS ,JOB_SPEC_GROUP_ID , JOB_SPEC_GROUP_PO_ID FROM ODS_OCS_PROD_A1A2.ODS_V_JOB_SPEC_GROUP_H m
								union all
								select FOUP_ID ,PTOOL_ID ,CHAMBERS ,JOB_SPEC_GROUP_ID , JOB_SPEC_GROUP_PO_ID FROM ODS_OCS_PROD_A3.ODS_V_JOB_SPEC_GROUP_H n ) b
								where b.JOB_SPEC_GROUP_PO_ID =a.JOB_GROUP_ID
								AND a.PROCESS_EQUIP = 'CMA51'	-- 机台
								AND a.MEASURE_TIME >= DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND a.MEASURE_TIME < DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR) -- 时间段，客户时间前后各推3个小时
								) tmp
								GROUP BY FOUP_ID, FOUPSLOT_ID
							)
						AND INSPECTION_TIME >= DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND INSPECTION_TIME < DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR) -- 时间段，客户时间前后各推3个小时
						ORDER BY INSPECTION_TIME
					)
				AND isd.INSPECTION_TIME BETWEEN DATE_SUB('2023-10-05 04:11:00', INTERVAL 0 HOUR) AND DATE_ADD('2023-10-05 17:46:00', INTERVAL 3 HOUR) -- 时间段，客户时间前后各推3个小时
			) TMP1
			LEFT JOIN
			ODS_UDB.ODS_INSP_WAFER_SUMMARY TMP2
			ON TMP1.WAFER_KEY = TMP2.WAFER_KEY;