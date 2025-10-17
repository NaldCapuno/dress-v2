-- Dummy data for students and violations
-- Aligns with schema in dress_clean.sql
-- Tables: `students`(student_id, rfid_uid, name, gender, year_level, course, college)
--         `violations`(violation_id [AUTO], student_id, violation_type, timestamp [auto], image_proof, status)

-- Insert dummy students
INSERT INTO `students` (`student_id`, `rfid_uid`, `name`, `gender`, `year_level`, `course`, `college`) VALUES
  ('2020-1-0001', 'A1B2C3D4', 'Alexandra Reyes', 'female', 1, 'BS Computer Science', 'College of Sciences'),
  ('2021-2-0002', 'E5F6G7H8', 'Miguel Santos', 'male', 2, 'BS Biology', 'College of Sciences'),
  ('2022-3-0003', 'J9K0L1M2', 'Patricia Dela Cruz', 'female', 3, 'BS Mathematics', 'College of Sciences'),
  ('2023-4-0004', 'N3P4Q5R6', 'Joshua Lim', 'male', 4, 'BS Information Technology', 'College of Sciences'),
  ('2020-1-0005', 'S7T8U9V0', 'Rafael Garcia', 'male', 1, 'BS Civil Engineering', 'College of Engineering'),
  ('2021-2-0006', 'W1X2Y3Z4', 'Beatriz Mendoza', 'female', 2, 'BS Mechanical Engineering', 'College of Engineering'),
  ('2022-3-0007', 'H4J5K6L7', 'Carlo Navarro', 'male', 3, 'BS Electrical Engineering', 'College of Engineering'),
  ('2023-4-0008', 'M8N9P0Q1', 'Diana Villanueva', 'female', 4, 'BS Electronics Engineering', 'College of Engineering'),
  ('2021-1-0009', 'R2S3T4U5', 'Isabella Cruz', 'female', 1, 'BS Architecture', 'College of Architecture and Design'),
  ('2022-2-0010', 'V6W7X8Y9', 'Gabriel Perez', 'male', 2, 'BS Industrial Design', 'College of Architecture and Design'),
  ('2023-3-0011', 'Z0A1B2C3', 'Sofia Ramirez', 'female', 3, 'BS Architecture', 'College of Architecture and Design'),
  ('2024-4-0012', 'D4E5F6G7', 'Luis Fernandez', 'male', 4, 'BS Interior Design', 'College of Architecture and Design'),
  ('2024-1-0013', 'H8I9J0K1', 'Kimberly Ong', 'female', 1, 'BA English', 'College of Arts and Humanities'),
  ('2020-2-0014', 'L2M3N4O5', 'Nathaniel Chua', 'male', 2, 'BS Accountancy', 'College of Business and Accountancy'),
  ('2024-3-0015', 'P6Q7R8S9', 'Christine Tan', 'female', 3, 'BS Criminology', 'College of Criminal Justice Education'),
  ('2021-4-0016', 'T0U1V2W3', 'Jerome Bautista', 'male', 4, 'BS Hospitality Management', 'College of Hospitality Management and Tourism'),
  ('2020-3-0017', 'X4Y5Z6A7', 'Monica Alvarez', 'female', 3, 'BS Nursing', 'College of Nursing and Health Sciences'),
  ('2022-4-0018', 'B8C9D0E1', 'Andre Rodriguez', 'male', 4, 'BSEd Mathematics', 'College of Teacher Education');

-- Insert dummy violations (must reference existing student_id values)
INSERT INTO `violations` (`student_id`, `violation_type`, `image_proof`, `status`) VALUES
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


