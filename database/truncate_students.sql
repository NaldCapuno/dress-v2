-- Script to truncate students table without foreign key constraint errors
-- This temporarily disables foreign key checks, truncates the table, then re-enables checks

-- Disable foreign key checks
SET FOREIGN_KEY_CHECKS = 0;

-- Truncate the students table
TRUNCATE TABLE `students`;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- Optional: Also truncate related tables if you want to clear everything
-- Uncomment the lines below if needed:

-- TRUNCATE TABLE `rfid_logs`;
-- TRUNCATE TABLE `violations`;

SELECT 'Students table truncated successfully!' AS result;

