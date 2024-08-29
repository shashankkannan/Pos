# POS System - Python Django Web Application

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Django](https://img.shields.io/badge/Django-3.x-brightgreen.svg)](https://www.djangoproject.com/)

## Overview

This project is a Point of Sale (POS) system developed using Django, a powerful Python web framework. The system allows users to scan items, add them to a shopping cart, and proceed to checkout. It includes essential features such as calculating taxes, managing cart items, and processing payments.

## Features

- **Scan Portal:** Easily add items to the cart by scanning barcodes.
- **Cart Management:** View items in the cart, along with their prices and images.
- **Tax Calculation:** Automatically calculates a 10% tax on the total cart value.
- **Checkout Process:** View cart items on the checkout page, with an option to finalize the purchase.
- **Payment Gateway Simulation:** A pop-up box simulates a payment process, asking for card details.
- **Session Management:** Cart items are stored in session, allowing for persistence during the shopping process.
- **Cart Clearance:** Automatically clears the cart once the checkout process is completed.

## Project Structure

```bash
pos/
├── myapp/                  # Main application directory
│   ├── migrations/         # Django migrations
│   ├── static/             # Static files (images, CSS, JavaScript)
│   ├── templates/          # HTML templates
│   │   └── myapp/
│   │       ├── checkout.html
│   │       └── ... 
│   ├── views.py            # Views handling requests
│   └── ...
├── pos/                    # Project configuration directory
├── db.sqlite3              # SQLite3 database
├── manage.py               # Django management script
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.7+
- Django 3.x

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shashankkannan/Pos.git
   cd Pos
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Migrations:**
   ```bash
   python manage.py migrate
   ```

4. **Start the Development Server:**
   ```bash
   python manage.py runserver
   ```

5. **Access the Application:**
   Open your web browser and navigate to `http://127.0.0.1:8000/` to start using the POS system.

## Usage

- **Scanning Items:** Use the scan portal to add items to the cart.
- **Cart Overview:** Review your cart with item images, names, prices, and calculated tax.
- **Checkout:** Click on the checkout button to view the cart summary and proceed to payment.
- **Payment Process:** Enter card details in the pop-up box to simulate payment.

## Contact

For any questions or inquiries, please contact me at shashank.kannan.cs@gmail.com.
