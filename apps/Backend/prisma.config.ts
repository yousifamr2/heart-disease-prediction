import "dotenv/config";
import path from "path";
import { defineConfig } from "prisma/config";

// تحميل .env من نفس مجلد المشروع
import { config } from "dotenv";
config({ path: path.join(__dirname, ".env") });

export default defineConfig({
  schema: "prisma/schema.prisma",
  migrations: {
    path: "prisma/migrations",
  },
  datasource: {
    url: process.env["DATABASE_URL"]!,
  },
});
