CREATE TABLE insuranceMetrics (
    id BIGSERIAL PRIMARY KEY, -- Automatically generates a unique value for each row
    createdAt TIMESTAMP,
    targetColumnDriftScore FLOAT,
    driftColumnShare FLOAT,
    jensenSex FLOAT,
    jensenSmoker FLOAT,
    jensenChildren FLOAT,
    jensenBmi FLOAT,
    jensenAge FLOAT,
    bmiMean FLOAT,
    ageMean FLOAT,
    mostCommonSex VARCHAR,
    mostCommonSmoker VARCHAR,
    mostCommonChildren VARCHAR,
    missingShareCurrent FLOAT,
    missingShareReference FLOAT
);

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY, -- Auto-generated unique identifier
    requestid BIGINT,         -- Column to store request ID
    sex VARCHAR(10),          -- Column for sex as a string
    children VARCHAR(10),     -- Column for children as a string
    smoker VARCHAR(10),       -- Column for smoker status as a string
    age INTEGER,              -- Column for age as an integer
    bmi FLOAT,                -- Column for BMI as a float
    charges FLOAT             -- Column for charges as a float
);