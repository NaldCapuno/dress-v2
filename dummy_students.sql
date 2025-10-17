-- Dummy data for students table
-- Fields: student_id, rfid_uid, name, gender, year_level, course, college

INSERT INTO `students` (`student_id`, `rfid_uid`, `name`, `gender`, `year_level`, `course`, `college`)
VALUES
  -- College of Sciences
  ('2020-1-0001', 'A1B2C3D4', 'Alexandra Reyes', 'female', 1, 'BS Computer Science', 'College of Sciences'),
  ('2021-2-0002', 'E5F6G7H8', 'Miguel Santos', 'male', 2, 'BS Biology', 'College of Sciences'),
  ('2022-3-0003', 'J9K0L1M2', 'Patricia Dela Cruz', 'female', 3, 'BS Mathematics', 'College of Sciences'),
  ('2023-4-0004', 'N3P4Q5R6', 'Joshua Lim', 'male', 4, 'BS Information Technology', 'College of Sciences'),

  -- College of Engineering
  ('2020-1-0005', 'S7T8U9V0', 'Rafael Garcia', 'male', 1, 'BS Civil Engineering', 'College of Engineering'),
  ('2021-2-0006', 'W1X2Y3Z4', 'Beatriz Mendoza', 'female', 2, 'BS Mechanical Engineering', 'College of Engineering'),
  ('2022-3-0007', 'H4J5K6L7', 'Carlo Navarro', 'male', 3, 'BS Electrical Engineering', 'College of Engineering'),
  ('2023-4-0008', 'M8N9P0Q1', 'Diana Villanueva', 'female', 4, 'BS Electronics Engineering', 'College of Engineering'),

  -- College of Architecture and Design
  ('2021-1-0009', 'R2S3T4U5', 'Isabella Cruz', 'female', 1, 'BS Architecture', 'College of Architecture and Design'),
  ('2022-2-0010', 'V6W7X8Y9', 'Gabriel Perez', 'male', 2, 'BS Industrial Design', 'College of Architecture and Design'),
  ('2023-3-0011', 'Z0A1B2C3', 'Sofia Ramirez', 'female', 3, 'BS Architecture', 'College of Architecture and Design'),
  ('2024-4-0012', 'D4E5F6G7', 'Luis Fernandez', 'male', 4, 'BS Interior Design', 'College of Architecture and Design'),

  -- More Sciences
  ('2024-1-0013', 'H8I9J0K1', 'Kimberly Ong', 'female', 1, 'BS Chemistry', 'College of Sciences'),
  ('2020-2-0014', 'L2M3N4O5', 'Nathaniel Chua', 'male', 2, 'BS Physics', 'College of Sciences'),

  -- More Engineering
  ('2024-3-0015', 'P6Q7R8S9', 'Christine Tan', 'female', 3, 'BS Computer Engineering', 'College of Engineering'),
  ('2021-4-0016', 'T0U1V2W3', 'Jerome Bautista', 'male', 4, 'BS Industrial Engineering', 'College of Engineering'),

  -- More Architecture and Design
  ('2020-3-0017', 'X4Y5Z6A7', 'Monica Alvarez', 'female', 3, 'BS Architecture', 'College of Architecture and Design'),
  ('2022-4-0018', 'B8C9D0E1', 'Andre Rodriguez', 'male', 4, 'BS Landscape Architecture', 'College of Architecture and Design');


-- Dummy data for violations table
-- Fields: student_id, violation_type, image_proof, status
INSERT INTO `violations` (`student_id`, `violation_type`, `image_proof`, `status`)
VALUES
  ('2020-1-0001', 'missing polo_shirt', NULL, 'pending'),
  ('2021-2-0002', 'missing pants', 'uploads/violations/2021-2-0002_no_id.jpg', 'forwarded_dean'),
  ('2022-3-0003', 'missing shoes', NULL, 'pending'),
  ('2023-4-0004', 'missing blouse', 'uploads/violations/2023-4-0004_torn.jpg', 'resolved'),
  ('2020-1-0005', 'missing skirt', NULL, 'pending'),
  ('2021-2-0006', 'missing doll_shoes', NULL, 'forwarded_guidance'),
  ('2022-3-0007', 'missing polo_shirt', 'uploads/violations/2022-3-0007_haircut.jpg', 'pending'),
  ('2023-4-0008', 'missing pants', NULL, 'pending'),
  ('2021-1-0009', 'missing shoes', 'uploads/violations/2021-1-0009_skirt.jpg', 'forwarded_dean'),
  ('2022-2-0010', 'missing blouse', NULL, 'pending'),
  ('2023-3-0011', 'missing skirt', NULL, 'resolved'),
  ('2024-4-0012', 'missing doll_shoes', NULL, 'pending'),
  ('2024-1-0013', 'missing polo_shirt', 'uploads/violations/2024-1-0013_no_id.jpg', 'pending'),
  ('2020-2-0014', 'missing pants', NULL, 'forwarded_guidance'),
  ('2024-3-0015', 'missing shoes', NULL, 'pending'),
  ('2021-4-0016', 'missing blouse', 'uploads/violations/2021-4-0016_uniform.jpg', 'pending'),
  ('2020-3-0017', 'missing skirt', NULL, 'pending'),
  ('2022-4-0018', 'missing doll_shoes', NULL, 'pending');


-- Additional Science students to test dean alerts (College of Sciences)
INSERT INTO `students` (`student_id`, `rfid_uid`, `name`, `gender`, `year_level`, `course`, `college`)
VALUES
  ('2025-1-0020', 'SCI-UID-20', 'Erika Santos', 'female', 1, 'BS Computer Science', 'College of Sciences'),
  ('2025-1-0021', 'SCI-UID-21', 'Leo Ramirez', 'male', 2, 'BS Biology', 'College of Sciences'),
  ('2025-1-0022', 'SCI-UID-22', 'Paolo Cruz', 'male', 3, 'BS Mathematics', 'College of Sciences'),
  ('2025-1-0023', 'SCI-UID-23', 'Hannah Tan', 'female', 4, 'BS Information Technology', 'College of Sciences');

-- Add pending violations older than 3 days for >=10 Science students
-- Note: explicit timestamp ensures they are counted by the alert checker
INSERT INTO `violations` (`student_id`, `violation_type`, `timestamp`, `image_proof`, `status`)
VALUES
  ('2020-1-0001', 'missing polo_shirt', '2025-10-01 08:00:00', NULL, 'pending'),
  ('2021-2-0002', 'missing pants',      '2025-10-01 08:05:00', NULL, 'pending'),
  ('2022-3-0003', 'missing shoes',      '2025-10-01 08:10:00', NULL, 'pending'),
  ('2023-4-0004', 'missing blouse',     '2025-10-01 08:15:00', NULL, 'pending'),
  ('2024-1-0013', 'missing skirt',      '2025-10-01 08:20:00', NULL, 'pending'),
  ('2020-2-0014', 'missing doll_shoes', '2025-10-01 08:25:00', NULL, 'pending'),
  ('2025-1-0020', 'missing polo_shirt', '2025-10-01 08:30:00', NULL, 'pending'),
  ('2025-1-0021', 'missing pants',      '2025-10-01 08:35:00', NULL, 'pending'),
  ('2025-1-0022', 'missing shoes',      '2025-10-01 08:40:00', NULL, 'pending'),
  ('2025-1-0023', 'missing blouse',     '2025-10-01 08:45:00', NULL, 'pending');


-- More College of Sciences students to further test dean alerts
INSERT INTO `students` (`student_id`, `rfid_uid`, `name`, `gender`, `year_level`, `course`, `college`)
VALUES
  ('2025-1-0024', 'SCI-UID-24', 'Janelle Uy', 'female', 2, 'BS Biology', 'College of Sciences'),
  ('2025-1-0025', 'SCI-UID-25', 'Karl Dizon', 'male', 3, 'BS Mathematics', 'College of Sciences'),
  ('2025-1-0026', 'SCI-UID-26', 'Lara Go', 'female', 1, 'BS Chemistry', 'College of Sciences'),
  ('2025-1-0027', 'SCI-UID-27', 'Mark Co', 'male', 4, 'BS Information Technology', 'College of Sciences'),
  ('2025-1-0028', 'SCI-UID-28', 'Nina Sy', 'female', 2, 'BS Computer Science', 'College of Sciences');

-- Pending violations older than 3 days for the additional Science students
INSERT INTO `violations` (`student_id`, `violation_type`, `timestamp`, `image_proof`, `status`)
VALUES
  ('2025-1-0024', 'missing pants',      '2025-09-28 09:00:00', NULL, 'pending'),
  ('2025-1-0025', 'missing shoes',      '2025-09-28 09:05:00', NULL, 'pending'),
  ('2025-1-0026', 'missing polo_shirt', '2025-09-28 09:10:00', NULL, 'pending'),
  ('2025-1-0027', 'missing blouse',     '2025-09-28 09:15:00', NULL, 'pending'),
  ('2025-1-0028', 'missing skirt',      '2025-09-28 09:20:00', NULL, 'pending');

