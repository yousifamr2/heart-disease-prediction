-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "national_id" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "labs" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "lab_code" TEXT NOT NULL,
    "address" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "labs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "lab_tests" (
    "id" TEXT NOT NULL,
    "lab_id" TEXT NOT NULL,
    "national_id" TEXT NOT NULL,
    "age" DOUBLE PRECISION NOT NULL,
    "sex" INTEGER NOT NULL,
    "chest_pain_type" INTEGER NOT NULL,
    "resting_bp_s" DOUBLE PRECISION NOT NULL,
    "cholesterol" DOUBLE PRECISION NOT NULL,
    "fasting_blood_sugar" INTEGER NOT NULL,
    "resting_ecg" INTEGER NOT NULL,
    "max_heart_rate" DOUBLE PRECISION NOT NULL,
    "exercise_angina" INTEGER NOT NULL,
    "oldpeak" DOUBLE PRECISION NOT NULL,
    "st_slope" INTEGER NOT NULL,
    "prediction_result" TEXT,
    "prediction_percentage" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "lab_tests_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "hospitals" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "area" TEXT NOT NULL,
    "google_maps_link" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "hospitals_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_national_id_key" ON "users"("national_id");

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "labs_lab_code_key" ON "labs"("lab_code");

-- CreateIndex
CREATE INDEX "lab_tests_national_id_idx" ON "lab_tests"("national_id");

-- AddForeignKey
ALTER TABLE "lab_tests" ADD CONSTRAINT "lab_tests_lab_id_fkey" FOREIGN KEY ("lab_id") REFERENCES "labs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
