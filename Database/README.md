# Heart Disease Prediction API

Backend API for Heart Disease Prediction System using Node.js, Express, and MongoDB.

## Features

- ✅ JWT Authentication (Register/Login with tokens)
- ✅ Protected Routes (JWT middleware)
- ✅ User Management (CRUD)
- ✅ Lab Management (CRUD)
- ✅ Lab Test Management (CRUD)
- ✅ Hospital Management (CRUD)
- ✅ Password Hashing (bcrypt)
- ✅ Input Validation
- ✅ Pagination Support
- ✅ Error Handling
- ✅ Request Logging
- ✅ Security Headers (Helmet)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Update `.env` with your MongoDB connection string and JWT secret:
```
MONGO_URI=mongodb+srv://username:password@cluster0.yal67jf.mongodb.net/database?retryWrites=true&w=majority
PORT=5000
NODE_ENV=development
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRE=30d
```

**Important:** Use a strong, random string for `JWT_SECRET` in production!

4. Seed labs data (optional):
```bash
npm run seed:labs
```

5. Start the server:
```bash
# Development mode (with nodemon)
npm run dev

# Production mode
npm start
```

## API Documentation

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for complete API reference.

## Project Structure

```
Database/
├── config/
│   └── db.js              # Database connection
├── middleware/
│   ├── validation.js      # Input validation middleware
│   └── pagination.js      # Pagination middleware
├── models/
│   ├── user.js            # User model
│   ├── lab.js             # Lab model
│   ├── labtest.js         # Lab Test model
│   └── hospital.js        # Hospital model
├── routes/
│   ├── authRoute.js       # Authentication routes
│   ├── userRoute.js       # User routes
│   ├── labRoute.js        # Lab routes
│   ├── labtestRoute.js    # Lab Test routes
│   └── hospitalRoute.js   # Hospital routes
├── index.js               # Main server file
├── seedLabs.js            # Seed script for labs
└── package.json           # Dependencies
```

## API Endpoints Summary

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user

### Users
- `GET /api/users` - Get all users (paginated)
- `GET /api/users/:id` - Get user by ID
- `POST /api/users` - Create user
- `PUT /api/users/:id` - Update user
- `DELETE /api/users/:id` - Delete user

### Labs
- `GET /api/labs` - Get all labs (paginated)
- `GET /api/labs/:id` - Get lab by ID
- `POST /api/labs` - Create lab
- `PUT /api/labs/:id` - Update lab
- `DELETE /api/labs/:id` - Delete lab

### Lab Tests
- `GET /api/labtests` - Get all lab tests (paginated)
- `GET /api/labtests/:id` - Get lab test by ID
- `GET /api/labtests/patient/:national_id` - Get all tests for patient
- `GET /api/labtests/patient/:national_id/latest` - Get latest test for patient
- `GET /api/labtests/lab/:lab_id` - Get tests by lab
- `POST /api/labtests` - Create lab test
- `PUT /api/labtests/:id` - Update lab test
- `PATCH /api/labtests/:id/prediction` - Update prediction only
- `DELETE /api/labtests/:id` - Delete lab test

### Hospitals
- `GET /api/hospitals` - Get all hospitals (paginated)
- `GET /api/hospitals/:id` - Get hospital by ID
- `GET /api/hospitals/area/:area` - Get hospitals by area
- `POST /api/hospitals` - Create hospital
- `PUT /api/hospitals/:id` - Update hospital
- `DELETE /api/hospitals/:id` - Delete hospital

## Authentication

This API uses JWT (JSON Web Tokens) for authentication.

1. **Register/Login** to get a token:
   ```bash
   POST /api/auth/register
   POST /api/auth/login
   ```

2. **Use token** in protected routes:
   ```
   Authorization: Bearer <your-token>
   ```

3. **Protected Routes** require authentication:
   - All CREATE, UPDATE, DELETE operations
   - GET operations are public (no token needed)

## Technologies Used

- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM for MongoDB
- **bcrypt** - Password hashing
- **jsonwebtoken** - JWT authentication
- **Helmet** - Security headers
- **dotenv** - Environment variables

## License

ISC
