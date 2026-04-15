const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });

const prisma = require("./config/prisma");

const labs = [
  {
    name: "Al Mokhtabar labs",
    lab_code: "Al Mokhtabar 123",
    address: "Alexandria , 228 Port Said Street, Zamzam Tower - Sporting",
  },
  {
    name: "AL Borg Labs",
    lab_code: "AL Borg 123",
    address: "Alexandria , 14 Faculty Of Medicine Street, Raml Station",
  },
  {
    name: "Hassab Labs",
    lab_code: "Hassab 123",
    address: "Alexandria , 405 Al Horreya Road, Abu Qir Street - Sidi Gaber",
  },
  {
    name: "Royal Labs",
    lab_code: "Royal 123",
    address: "Alexandria , 36 Saad Zaghloul Street, Raml Station, above Chicorel, 3rd floor",
  },
  {
    name: "Al Shams Labs",
    lab_code: "Al Shams 123",
    address: "Alexandria , 86 Moharram Bek Street, in Front Of The Old Awlad El Sheikh Mosque",
  },
  {
    name: "Al Nile Labs",
    lab_code: "Al Nile 123",
    address: "59 Safia Zaghloul Street, Raml Station, Alex Tower Commercial Building - Raml Station",
  },
];

const seedLabs = async () => {
  try {
    await prisma.$connect();
    console.log("PostgreSQL (Neon) Connected Successfully!");

    await prisma.lab.deleteMany({});
    console.log("Existing labs deleted");

    const created = await prisma.lab.createMany({ data: labs });
    console.log(`${created.count} labs seeded successfully!`);

    const all = await prisma.lab.findMany({ orderBy: { createdAt: "asc" } });
    all.forEach((lab, i) =>
      console.log(`${i + 1}. ${lab.name} — Code: ${lab.lab_code} — ID: ${lab.id}`)
    );

    process.exit(0);
  } catch (err) {
    console.error("Error seeding labs:", err.message);
    process.exit(1);
  }
};

seedLabs();
