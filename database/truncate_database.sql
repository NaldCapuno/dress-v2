-- Script to truncate students table without foreign key constraint errors
-- This temporarily disables foreign key checks, truncates the table, then re-enables checks

-- Disable foreign key checks
SET FOREIGN_KEY_CHECKS = 0;

-- Truncate the admins table
TRUNCATE TABLE `admins`;

-- Truncate the email_outbox table
TRUNCATE TABLE `email_outbox`;

-- Truncate the rfid_logs table
TRUNCATE TABLE `rfid_logs`;

-- Truncate the settings table
TRUNCATE TABLE `settings`;

-- Truncate the students table
TRUNCATE TABLE `students`;

-- Truncate the violations table
TRUNCATE TABLE `violations`;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- Optional: Also truncate related tables if you want to clear everything
-- Uncomment the lines below if needed:

-- TRUNCATE TABLE `rfid_logs`;
-- TRUNCATE TABLE `violations`;

SELECT 'Truncated successfully!' AS result;

