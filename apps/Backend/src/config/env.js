const { z } = require("zod");

const envSchema = z.object({
  PORT: z.string().transform((val) => Number(val)).default("5000"),
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  DATABASE_URL: z.string().url("DATABASE_URL must be a valid connection string"),
  JWT_SECRET: z.string().min(8, "JWT_SECRET must be at least 8 characters long"),
  JWT_EXPIRE: z.string().default("30d"),
  ADMIN_API_KEY: z.string().min(1, "ADMIN_API_KEY is required"),
  LAB_API_KEY: z.string().default("admin-key-change-me"),
  AI_SERVICE_URL: z.string().url().default("http://127.0.0.1:8000"),
  INTERNAL_API_KEY: z.string().min(1, "INTERNAL_API_KEY is required"),
});

const parseEnv = () => {
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    console.error("❌ Invalid environment variables:");
    console.error(JSON.stringify(parsed.error.format(), null, 2));
    process.exit(1);
  }
  return parsed.data;
};

const env = parseEnv();

module.exports = env;
