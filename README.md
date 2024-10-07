# POS System - Python Django Web Application
**Upgrade**: Integrated POS and E-commerce Solution with AI-Powered Analysis - Project upgrade

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Django](https://img.shields.io/badge/Django-3.x-brightgreen.svg)](https://www.djangoproject.com/)

## Overview

This project is a Point of Sale (POS) system developed using Django, a powerful Python web framework. The system allows users to scan items, add them to a shopping cart, and proceed to checkout. It includes essential features such as calculating taxes, managing cart items, and processing payments.

## Features
- **POS Scanning Portal**: Supports item scanning, cart management, payment processing, and automatically generates orders and receipts.
- **E-commerce Store**: Customers can browse, search, add items to cart, and place orders online, with automatic order and receipt generation.
- **Stock Management**: Tracks stock levels, predicts restock timing, and flags expiring items for quick action.
- **Customer Analysis**: Identifies purchase patterns, provides insights on customer behavior, and predicts future visits.
- **AI Review System**: Allows customers to leave reviews using text or voice input, performs sentiment analysis, assigns star ratings, and generates automated replies. It flags problematic reviews for ticket creation, which are then assigned to workers.
- **Worker Portal**: Workers manage assigned tickets with login options using face recognition or username/password. Performance analytics help admins assess employee productivity and assign tasks effectively.
- **Predictive Analytics**: Provides key insights on item performance, stock wastage, and customer engagement for better decision-making.

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
