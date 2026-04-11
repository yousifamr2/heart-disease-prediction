# Heart Disease Prediction API Documentation

## Base URL
```
http://localhost:5000/api
```

---

## Authentication (JWT)

This API uses JWT (JSON Web Tokens) for authentication. After registering or logging in, you will receive a token that must be included in the Authorization header for protected routes.

### How to Use JWT Token

1. **Register or Login** to get a token
2. **Include token in requests** to protected endpoints:
   ```
   Authorization: Bearer <your-token-here>
   ```

### Token Format
```
Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Protected Routes
The following operations require authentication:
- ✅ Create, Update, Delete operations (all resources)
- ✅ User management operations
- ✅ Lab Test creation and updates
- ❌ GET operations are public (no authentication required)

### Token Expiration
- Default: 30 days
- Configurable via `JWT_EXPIRE` in `.env`

---

## Authentication Endpoints

### 1. Register User
**POST** `/auth/register`

Register a new user.

**Request Body:**
```json
{
  "national_id": "12345678901234",
  "username": "Ahmed Ali",
  "email": "ahmed@example.com",
  "password": "password123"
}
```

**Response (201):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "_id": "...",
    "national_id": "12345678901234",
    "username": "Ahmed Ali",
    "email": "ahmed@example.com",
    "createdAt": "...",
    "updatedAt": "..."
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Note:** Save the `token` from the response. You'll need it for protected routes.

**Validation:**
- `national_id`: Required, exactly 14 digits
- `username`: Required, 2-50 characters
- `email`: Required, valid email format
- `password`: Required, minimum 6 characters

---

### 2. Login User
**POST** `/auth/login`

Login with email and password.

**Request Body:**
```json
{
  "email": "ahmed@example.com",
  "password": "password123"
}
```

**Response (200):**
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "_id": "...",
    "national_id": "12345678901234",
    "username": "Ahmed Ali",
    "email": "ahmed@example.com"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Note:** Save the `token` from the response. You'll need it for protected routes.

**Error (401):**
```json
{
  "success": false,
  "message": "Invalid email or password"
}
```

---

## User Endpoints

### 1. Get All Users
**GET** `/users?page=1&limit=10`

Get paginated list of users.

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 10)

**Response (200):**
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 50,
    "totalPages": 5
  }
}
```

---

### 2. Get User by ID
**GET** `/users/:id`

Get single user by ID.

**Response (200):**
```json
{
  "success": true,
  "data": {
    "_id": "...",
    "national_id": "12345678901234",
    "username": "Ahmed Ali",
    "email": "ahmed@example.com"
  }
}
```

**Error (404):**
```json
{
  "success": false,
  "message": "User not found"
}
```

---

### 3. Create User
**POST** `/users` 🔒 **Protected**

Create a new user for management purposes (admin/user management). This is **not** a public registration endpoint and **does not return a token**.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:** Same as `/auth/register`

**Response (201):**
```json
{
  "success": true,
  "data": {
    "_id": "...",
    "national_id": "12345678901234",
    "username": "Ahmed Ali",
    "email": "ahmed@example.com",
    "createdAt": "...",
    "updatedAt": "..."
  }
}
```

---

### 4. Update User
**PUT** `/users/:id` 🔒 **Protected**

Update user information. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "username": "Updated Name",
  "email": "newemail@example.com"
}
```

---

### 5. Delete User
**DELETE** `/users/:id` 🔒 **Protected**

Delete a user. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200):**
```json
{
  "success": true,
  "message": "User deleted"
}
```

---

## Lab Endpoints

### 1. Get All Labs
**GET** `/labs?page=1&limit=10`

Get paginated list of labs.

**Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "_id": "...",
      "name": "Al Mokhtabar labs",
      "lab_code": "Al Mokhtabar 123",
      "address": "Alexandria , 228 Port Said Street..."
    }
  ],
  "pagination": {...}
}
```

---

### 2. Get Lab by ID
**GET** `/labs/:id`

Get single lab by ID.

---

### 3. Create Lab
**POST** `/labs` 🔒 **Protected**

Create a new lab. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "name": "Lab Name",
  "lab_code": "LAB001",
  "address": "Lab Address"
}
```

---

### 4. Update Lab
**PUT** `/labs/:id` 🔒 **Protected**

Update lab information. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "name": "Updated Lab Name",
  "lab_code": "LAB001",
  "address": "Updated Lab Address"
}
```

---

### 5. Delete Lab
**DELETE** `/labs/:id` 🔒 **Protected**

Delete a lab. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

---

## Lab Test Endpoints

### 1. Get All Lab Tests
**GET** `/labtests?page=1&limit=10`

Get paginated list of lab tests with populated lab information.

**Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "_id": "...",
      "lab_id": {
        "_id": "...",
        "name": "Al Mokhtabar labs",
        "lab_code": "Al Mokhtabar 123"
      },
      "national_id": "12345678901234",
      "features": {
        "age": 63,
        "sex": 1,
        "chest_pain_type": 3,
        "resting_bp_s": 145,
        "cholesterol": 233,
        "fasting_blood_sugar": 1,
        "resting_ecg": 0,
        "max_heart_rate": 150,
        "exercise_angina": 0,
        "oldpeak": 2.3,
        "st_slope": 0
      },
      "prediction_result": "High Risk",
      "prediction_percentage": 85.5
    }
  ],
  "pagination": {...}
}
```

---

### 2. Get Lab Test by ID
**GET** `/labtests/:id`

Get single lab test by ID.

---

### 3. Create Lab Test
**POST** `/labtests` 🔒 **Protected**

Create a new lab test. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "lab_id": "507f1f77bcf86cd799439011",
  "national_id": "12345678901234",
  "features": {
    "age": 63,
    "sex": 1,
    "chest_pain_type": 3,
    "resting_bp_s": 145,
    "cholesterol": 233,
    "fasting_blood_sugar": 1,
    "resting_ecg": 0,
    "max_heart_rate": 150,
    "exercise_angina": 0,
    "oldpeak": 2.3,
    "st_slope": 0
  }
}
```

**Required Features:**
- `age`: Number
- `sex`: Number (0 or 1)
- `chest_pain_type`: Number
- `resting_bp_s`: Number
- `cholesterol`: Number
- `fasting_blood_sugar`: Number (0 or 1)
- `resting_ecg`: Number
- `max_heart_rate`: Number
- `exercise_angina`: Number (0 or 1)
- `oldpeak`: Number
- `st_slope`: Number

---

### 4. Get Latest Lab Test by National ID
**GET** `/labtests/patient/:national_id/latest`

Get the most recent lab test for a patient.

**Response (200):**
```json
{
  "success": true,
  "data": {
    "_id": "...",
    "lab_id": {...},
    "national_id": "12345678901234",
    "features": {...},
    "prediction_result": "High Risk",
    "prediction_percentage": 85.5
  }
}
```

---

### 5. Get All Lab Tests by National ID
**GET** `/labtests/patient/:national_id`

Get all lab tests for a patient, sorted by newest first.

---

### 6. Get Lab Tests by Lab ID
**GET** `/labtests/lab/:lab_id`

Get all lab tests for a specific lab.

---

### 7. Update Lab Test
**PUT** `/labtests/:id` 🔒 **Protected**

Update lab test information. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "lab_id": "507f1f77bcf86cd799439011",
  "national_id": "12345678901234",
  "features": {
    "age": 65,
    "sex": 1,
    "chest_pain_type": 2,
    "resting_bp_s": 140,
    "cholesterol": 220,
    "fasting_blood_sugar": 0,
    "resting_ecg": 1,
    "max_heart_rate": 155,
    "exercise_angina": 0,
    "oldpeak": 1.5,
    "st_slope": 1
  }
}
```

---

### 8. Update Prediction Result Only
**PATCH** `/labtests/:id/prediction` 🔒 **Protected**

Update only prediction result and percentage. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "prediction_result": "High Risk",
  "prediction_percentage": 85.5
}
```

**Prediction Values:**
- `prediction_result`: "High Risk" or "Low Risk"
- `prediction_percentage`: Number between 0 and 100

---

### 9. Delete Lab Test
**DELETE** `/labtests/:id` 🔒 **Protected**

Delete a lab test. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

---

## Hospital Endpoints

### 1. Get All Hospitals
**GET** `/hospitals?page=1&limit=10`

Get paginated list of hospitals.

**Response (200):**
```json
{
  "success": true,
  "data": [
    {
      "_id": "...",
      "name": "Hospital Name",
      "area": "Smouha",
      "google_maps_link": "https://maps.google.com/..."
    }
  ],
  "pagination": {...}
}
```

---

### 2. Get Hospital by ID
**GET** `/hospitals/:id`

---

### 3. Get Hospitals by Area
**GET** `/hospitals/area/:area`

Get hospitals filtered by area (e.g., "Smouha", "Sidi Gaber").

**Example:**
```
GET /hospitals/area/Smouha
```

---

### 4. Create Hospital
**POST** `/hospitals` 🔒 **Protected**

Create a new hospital. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "name": "Hospital Name",
  "area": "Smouha",
  "google_maps_link": "https://maps.google.com/..."
}
```

---

### 5. Update Hospital
**PUT** `/hospitals/:id` 🔒 **Protected**

Update hospital information. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "name": "Updated Hospital Name",
  "area": "Smouha",
  "google_maps_link": "https://maps.google.com/..."
}
```

---

### 6. Delete Hospital
**DELETE** `/hospitals/:id` 🔒 **Protected**

Delete a hospital. Requires authentication.

**Headers:**
```
Authorization: Bearer <token>
```

---

## Error Responses

All errors follow this format:

```json
{
  "success": false,
  "message": "Error message here"
}
```

### Common Status Codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request (validation errors)
- `401`: Unauthorized
- `404`: Not Found
- `500`: Internal Server Error

---

## Response Format

All successful responses follow this format:

```json
{
  "success": true,
  "data": {...},
  "message": "Optional message"
}
```

For paginated responses:

```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 100,
    "totalPages": 10
  }
}
```

---

## Notes

1. **Password Hashing**: User passwords are automatically hashed using bcrypt before saving.

2. **Password in Responses**: Passwords are never returned in API responses.

3. **National ID Validation**: Must be exactly 14 digits.

4. **Email Validation**: Must be a valid email format.

5. **Pagination**: All "Get All" endpoints support pagination via `page` and `limit` query parameters.

6. **Timestamps**: All models include `createdAt` and `updatedAt` timestamps.

7. **Populated Fields**: Lab tests include populated `lab_id` information by default.

---

## Testing

You can test the API using:
- **Postman**: Import the endpoints above
- **cURL**: Use curl commands
- **Thunder Client**: VS Code extension
- **Insomnia**: API client

### Example cURL:

```bash
# Register User
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "national_id": "12345678901234",
    "username": "Ahmed Ali",
    "email": "ahmed@example.com",
    "password": "password123"
  }'

# Login (save the token from response)
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "ahmed@example.com",
    "password": "password123"
  }'

# Get All Labs (public - no token needed)
curl http://localhost:5000/api/labs?page=1&limit=10

# Create Lab Test (protected - token required)
curl -X POST http://localhost:5000/api/labtests \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "lab_id": "507f1f77bcf86cd799439011",
    "national_id": "12345678901234",
    "features": {
      "age": 63,
      "sex": 1,
      "chest_pain_type": 3,
      "resting_bp_s": 145,
      "cholesterol": 233,
      "fasting_blood_sugar": 1,
      "resting_ecg": 0,
      "max_heart_rate": 150,
      "exercise_angina": 0,
      "oldpeak": 2.3,
      "st_slope": 0
    }
  }'

# Update User (protected - token required)
curl -X PUT http://localhost:5000/api/users/USER_ID \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "username": "Updated Name"
  }'
```

### Error Responses for Authentication:

**401 Unauthorized (No Token):**
```json
{
  "success": false,
  "message": "No token provided. Please provide a valid token."
}
```

**401 Unauthorized (Invalid Token):**
```json
{
  "success": false,
  "message": "Invalid token"
}
```

**401 Unauthorized (Expired Token):**
```json
{
  "success": false,
  "message": "Token expired"
}
```
