ALTER TABLE `stockblock`
    ADD COLUMN `rank` INT NOT NULL DEFAULT 0 AFTER `status`;