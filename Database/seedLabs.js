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
    name: "Central Medical Laboratory",
    lab_code: "CML001",
    address: "123 Main Street, Smouha, Alexandria"
  },
  {
    name: "Alexandria Diagnostic Center",
    lab_code: "ADC002",
    address: "456 Corniche Road, Sidi Gaber, Alexandria"
  },
  {
    name: "El-Maamoura Medical Lab",
    lab_code: "EMM003",
    address: "789 El-Maamoura Street, Alexandria"
  },
  {
    name: "Loran Clinical Laboratory",
    lab_code: "LCL004",
    address: "321 Loran Avenue, Alexandria"
  },
  {
    name: "Smouha Health Lab",
    lab_code: "SHL005",
    address: "654 Health Center Road, Smouha, Alexandria"
  },
  {
    name: "Sidi Gaber Diagnostic Lab",
    lab_code: "SGD006",
    address: "987 Sidi Gaber Square, Alexandria"
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

