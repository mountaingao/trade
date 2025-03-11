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
