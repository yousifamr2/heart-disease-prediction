const mongoose = require("mongoose");
const dotenv = require("dotenv");
const Lab = require("./models/lab");

dotenv.config();

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("MongoDB Connected Successfully!");
  } catch (err) {
    console.error("MongoDB Connection Error: ", err);
    process.exit(1);
  }
};

const labs = [
  {
    name: "Al Mokhtabar labs",
    lab_code: "Al Mokhtabar 123",
    address: "Alexandria , 228 Port Said Street, Zamzam Tower - Sporting"
  },
  {
    name: "AL Borg Labs",
    lab_code: "AL Borg 123",
    address: "Alexandria , 14 Faculty Of Medicine Street, Raml Station"
  },
  {
    name: "Hassab Labs",
    lab_code: "Hassab 123",
    address: "Alexandria , 405 Al Horreya Road, Abu Qir Street - Sidi Gaber"
  },
  {
    name: "Royal Labs",
    lab_code: "Royal 123",
    address: "Alexandria ,  36 Saad Zaghloul Street, Raml Station, above Chicorel, 3rd floor"
  },
  {
    name: "Al Shams Labs",
    lab_code: "Al Shams 123",
    address: "Alexandria , 86 Moharram Bek Street, in Front Of The Old Awlad El Sheikh Mosque  "
  },
  {
    name: "Al Nile Labs",
    lab_code: "Al Nile 123",
    address: "59 Safia Zaghloul Street, Raml Station, Alex Tower Commercial Building - Raml Station"
  }

];

const seedLabs = async () => {
  try {
    await connectDB();

    // Delete existing labs (optional - comment out if you want to keep existing data)
    await Lab.deleteMany({});
    console.log("Existing labs deleted");

    // Insert new labs
    const createdLabs = await Lab.insertMany(labs);
    console.log(`${createdLabs.length} labs seeded successfully!`);

    // Display seeded labs
    console.log("\nSeeded Labs:"); 
    createdLabs.forEach((lab, index) => {
      console.log(`${index + 1}. ${lab.name} - Code: ${lab.lab_code}`);
    });

    process.exit(0);
  } catch (err) {
    console.error("Error seeding labs:", err);
    process.exit(1);
  }
};

// Run seed function
seedLabs();

