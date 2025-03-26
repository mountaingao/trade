ALTER TABLE `stockblock`
    ADD COLUMN `rank` INT NOT NULL DEFAULT 0 AFTER `status`;



CREATE TABLE `stock_rating` (
                                `id` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
                                `symbol` VARCHAR(50) NOT NULL COLLATE 'utf8mb4_0900_ai_ci',
                                `stockname` VARCHAR(100) NULL DEFAULT NULL COLLATE 'utf8mb4_0900_ai_ci',
                                `recent_turnover` DECIMAL(18,4) NULL DEFAULT NULL,
                                `recent_increase` DECIMAL(18,4) NULL DEFAULT NULL,
                                `market_cap` DECIMAL(18,4) NULL DEFAULT NULL,
                                `amplitude` DECIMAL(18,4) NULL DEFAULT NULL,
                                `jgcyd` DECIMAL(18,4) NULL DEFAULT NULL,
                                `lspf` DECIMAL(18,4) NULL DEFAULT NULL,
                                `focus` DECIMAL(18,4) NULL DEFAULT NULL,
                                `desire_daily` DECIMAL(18,4) NULL DEFAULT NULL,
                                `dragon_tiger` DECIMAL(18,4) NULL DEFAULT NULL,
                                `news_analysis` DECIMAL(18,4) NULL DEFAULT NULL,
                                `estimated_turnover` DECIMAL(18,4) NULL DEFAULT NULL,
                                `total_score` DECIMAL(18,4) NULL DEFAULT NULL,
                                `avg_jgcyd` DECIMAL(18,4) NULL DEFAULT NULL,
                                `avg_lspf` DECIMAL(18,4) NULL DEFAULT NULL,
                                `avg_focus` DECIMAL(18,4) NULL DEFAULT NULL,
                                `last_desire_daily` DECIMAL(18,4) NULL DEFAULT NULL,
                                `free_float_value` DECIMAL(18,4) NULL DEFAULT NULL,
                                `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
                                PRIMARY KEY (`symbol`) USING BTREE,
                                UNIQUE INDEX `id` (`id`) USING BTREE
)
    COLLATE='utf8mb4_0900_ai_ci'
ENGINE=InnoDB
;

ALTER TABLE `stock_rating`
    ADD COLUMN `rating_date` DATE NULL DEFAULT NULL AFTER `recent_turnover`;
SELECT `DEFAULT_COLLATION_NAME` FROM `information_schema`.`SCHEMATA` WHERE `SCHEMA_NAME`='trade';



ALTER TABLE `stockblock`
    CHANGE COLUMN `rank` `ranking` INT(10) NOT NULL DEFAULT '0' AFTER `status`;

CREATE TABLE `stock_info` (
                              `code` varchar(12) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '编码',
                              `name` varchar(32) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '名称',
                              `retained_profits` decimal(18,2) DEFAULT NULL COMMENT '净利润',
                              `total_value` decimal(18,2) DEFAULT NULL COMMENT '总市值',
                              `market_value` decimal(18,2) DEFAULT NULL COMMENT '流通市值',
                              `industry` varchar(32) COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '所处行业',
                              `dynamic_pe` decimal(15,2) DEFAULT NULL COMMENT '市盈率(动态)',
                              `pb` decimal(15,2) DEFAULT NULL COMMENT '市净率',
                              `roe` decimal(15,2) DEFAULT NULL COMMENT '净资产收益率，它是衡量公司盈利能力的一个重要财务指标，表示公司利用股东权益（净资产）创造利润的能力，最好在同行业内进行比较',
                              `margin_rate` decimal(15,2) DEFAULT NULL COMMENT '毛利率',
                              `net_profit_rate` decimal(15,2) DEFAULT NULL COMMENT '净利率',
                              PRIMARY KEY (`code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


CREATE TABLE `gp_base_info` (
                                `code` varchar(12) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '编码',
                                `dt` date NOT NULL COMMENT '日期',
                                `open` decimal(15,2) DEFAULT NULL COMMENT '开盘价',
                                `close` decimal(15,2) DEFAULT NULL COMMENT '收盘价',
                                `up` decimal(15,2) DEFAULT NULL COMMENT '涨跌',
                                `upr` decimal(15,2) DEFAULT NULL COMMENT '涨幅百分比',
                                `low` decimal(15,2) DEFAULT NULL COMMENT '最低价',
                                `high` decimal(15,2) DEFAULT NULL COMMENT '最高价',
                                `vol` bigint DEFAULT NULL COMMENT '成交量',
                                `vola` decimal(15,2) DEFAULT NULL COMMENT '成交额',
                                `tr` decimal(15,2) DEFAULT NULL COMMENT '换手率，百分比',
                                PRIMARY KEY (`code`,`dt`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;



ALTER TABLE `alertdata`
    ADD COLUMN `score` FLOAT NOT NULL DEFAULT 0 AFTER `price_change`,
	ADD COLUMN `popup_status` TINYINT NOT NULL DEFAULT 0 AFTER `score`,
	CHANGE COLUMN `status` `status` VARCHAR(20) NOT NULL COLLATE 'utf8mb4_unicode_ci' AFTER `popup_status`;
