CREATE DATABASE laguna_crop_yield;
USE laguna_crop_yield;

CREATE TABLE rice_field (
    ID_rice INT AUTO_INCREMENT PRIMARY KEY,
    municipality VARCHAR(50),
    year INT,
    season TINYINT CHECK (season IN (1, 2))  -- Ensuring valid season values
);

CREATE TABLE real_time (
    ID_real INT AUTO_INCREMENT PRIMARY KEY,
    ID_rice INT NOT NULL,  -- Ensure it matches rice_field.ID_rice
    FOREIGN KEY (ID_rice) REFERENCES rice_field(ID_rice) ON DELETE CASCADE,
    date DATE,
    phase TINYINT CHECK (phase IN (1, 2, 3)),  -- Fixed the column name
    yield FLOAT
);


CREATE TABLE historical (
    ID_his INT AUTO_INCREMENT PRIMARY KEY,
    ID_rice INT NOT NULL,  -- Ensuring it cannot be NULL
    FOREIGN KEY (ID_rice) REFERENCES rice_field(ID_rice) ON DELETE CASCADE,
    yield FLOAT
);

CREATE TEMPORARY TABLE temp_historical (
    municipality VARCHAR(50),
    year INT,
    season TINYINT,
    yield FLOAT
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/historical_data.csv'
INTO TABLE temp_historical
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(municipality, year, season, yield);


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/historical_data.csv'
INTO TABLE rice_field
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Municipality, Year, Season, @dummy);

INSERT INTO historical (ID_rice, yield)
SELECT rf.ID_rice, th.yield
FROM temp_historical th
JOIN rice_field rf
ON th.municipality = rf.municipality
AND th.year = rf.year
AND th.season = rf.season;

# DROP TEMPORARY TABLE temp_historical;

# DROP DATABASE laguna_crop_yield;
 
SELECT * FROM real_time;










a



