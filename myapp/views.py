import base64
import os
from io import BytesIO
from pathlib import Path
from .creds import email, password
import openpyxl
import qrcode
from django.contrib import messages
from django.core.files import File
import pandas as pd
import re
from .tests import generate_response
import uuid
from collections import defaultdict
from decimal import Decimal
from urllib.parse import urlencode
import pyttsx3
import speech_recognition as sr
from PIL import Image
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.db.models import F, Func, Value
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest, HttpResponseServerError
import json
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.generic import ListView

from . import settings
from .models import Item, Type, Order, Receipt, Review, Ticket, Customer, worker
from .forms import ItemForm, TypeForm, WorkerLoginForm, UploadFileForm
import random
import cv2
import numpy as np

from .sentiment_analyzer import return_sentiment

from .forms import WorkerRegistrationForm
from imgbeddings import imgbeddings  # Make sure to install and import the imgbeddings package

import cv2

IMAGE_SAVE_PATH = os.path.join(settings.MEDIA_ROOT, 'captured_images')

from datetime import datetime, timedelta
import ast
from hugchat import hugchat
from hugchat.login import Login
import requests


def worker_analytics(request, worker_id):
    workers = worker.objects.get(worker_id=worker_id)
    tickets = workers.tickets.all()

    ticket_data = []
    for ticket in tickets:
        ticket_data.append({
            'ticket_id': ticket.ticket_id,
            'created_at': ticket.created_at,
            'done': ticket.done
        })

    return JsonResponse({
        'tickets': ticket_data
    })


def item_analytics(request, item_id):
    # Retrieve the item based on the provided item_id
    item = get_object_or_404(Item, id=item_id)

    # Get all orders related to this item
    orders = Order.objects.filter(item=item).values('created_at', 'quantity')

    # Calculate total orders and unique customers
    total_orders = orders.count()
    unique_customers = orders.values('customer').distinct().count()

    # Retrieve restock data and convert from string to dictionary
    restock_data_str = item.restock_dates  # This is the field in your model
    restock_data = ast.literal_eval(restock_data_str) if restock_data_str else {}

    # Calculate total wasted stock from restock data
    total_wasted_stock = 0
    formatted_restock_data = {}
    for date, (old_stock, new_stock, throwing_out) in restock_data.items():
        try:
            # Convert date string to datetime object and format it to 'YYYY-MM-DD'
            date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')  # Modify the format as per your date string
            formatted_date = date_obj.strftime('%Y-%m-%d')  # Format to 'YYYY-MM-DD'
        except ValueError:
            formatted_date = date
        if throwing_out:
            total_wasted_stock += old_stock  # Sum up old stock that's thrown out
        formatted_restock_data[formatted_date] = [old_stock, new_stock, bool(throwing_out)]
    restock_data_json = json.dumps(formatted_restock_data)

    order_dates = [order['created_at'].strftime('%Y-%m-%d') for order in orders]
    order_quantities = [order['quantity'] for order in orders]

    # Prepare data for customer orders graph
    customer_data = defaultdict(list)
    o = Order.objects.filter(item=item_id).values('email_id', 'created_at', 'quantity')
    for order in o:
        customer_email = order['email_id']
        customer_data[customer_email].append((order['created_at'].strftime('%Y-%m-%d'), order['quantity']))

    # Prepare data for the graph
    customer_orders = {
        customer: sorted(data)  # Sort by date
        for customer, data in customer_data.items()
    }

    customer_order_dates = {}
    customer_order_quantities = {}
    for customer, data in customer_orders.items():
        dates, quantities = zip(*data) if data else ([], [])
        customer_order_dates[customer] = list(dates)
        customer_order_quantities[customer] = list(quantities)

    l = list(formatted_restock_data.keys())[-1]
    last_date = datetime.strptime(l, '%Y-%m-%d')
    expiry_date = last_date + timedelta(days=item.expiry_days)
    print(last_date)
    # Prepare the context to pass to the template
    context = {
        'item': item,
        'total_orders': total_orders,
        'unique_customers': unique_customers,
        'restock_data': restock_data_json,
        'rd': formatted_restock_data,
        'total_wasted_stock': total_wasted_stock,
        'order_dates': json.dumps(order_dates),
        'order_quantities': json.dumps(order_quantities),
        'customer_order_dates': json.dumps(customer_order_dates),
        'customer_order_quantities': json.dumps(customer_order_quantities),
        'last_date': last_date,
        'expiry_date': expiry_date.strftime('%Y-%m-%d')
    }

    # Render the analytics template
    return render(request, 'myapp/item_analytics.html', context)


def upload_excel(request):
    if request.method == 'POST':
        excel_file = request.FILES['excel_file']

        wb = openpyxl.load_workbook(excel_file)
        sheet = wb.active

        # Assuming the columns of the Excel file match your expected format
        for row in sheet.iter_rows(min_row=2, values_only=True):
            item_name = row[0]
            quantity = row[1]
            created_at = row[2]
            email_id = row[3]

            # Get the Item object
            try:
                item = Item.objects.get(name=item_name)
            except Item.DoesNotExist:
                messages.error(request, f"Item '{item_name}' does not exist.")
                continue

            # Get or create the Customer object
            customer, created = Customer.objects.get_or_create(email=email_id)

            # Create an Order
            order = Order.objects.create(
                item=item,
                quantity=quantity,
                created_at=created_at,
                email_id=email_id,
                customer=customer,
            )

            # Calculate receipt values (tax and total)
            total = order.quantity * item.price
            tax = Decimal(0.10) * total

            # Create the Receipt
            receipt = Receipt.objects.create(
                total_tax=tax,
                total_bill=total + tax,
                created_at=created_at
            )
            receipt.orders.add(order)
            customer.total_spent = Decimal(str(customer.total_spent)) + Decimal(total + tax)
            customer.save()

        messages.success(request, "Excel file processed and orders/receipts created.")
        return redirect('upload_excel')  # Redirect to a success page or the upload page again

    # Render the upload form
    return render(request, 'myapp/upload_excel.html')


@csrf_exempt  # Use this only if you are not handling CSRF tokens correctly in your AJAX calls
@require_POST
def restock_item(request):
    data = json.loads(request.body)
    item_id = data.get('id')  # Get the item ID from the request
    quantity = data.get('quantity')
    is_throwing_out = data.get('isThrowingOut', False)
    if is_throwing_out is None:
        is_throwing_out = False
    if isinstance(quantity, int) and quantity >= 0:
        try:
            # Retrieve the item from the database using the item ID
            item = Item.objects.get(id=item_id)
            oq = item.stock
            # Update the stock quantity
            item.stock = quantity

            # Handle the restock_dates field
            if item.restock_dates:
                # Convert existing restock_dates string to a dictionary
                restock_dates_dict = ast.literal_eval(item.restock_dates)  # Safely evaluate string as dictionary
            else:
                # Initialize an empty dictionary if no restock_dates are present
                restock_dates_dict = {}

            # Create the datetime key with the current datetime
            datetime_key = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            restock_dates_dict[datetime_key] = (
                oq, quantity, is_throwing_out)  # Store as tuple (current stock, restock amount)

            # Convert the dictionary back to a string to store in the model
            item.restock_dates = str(restock_dates_dict)
            item.available = True
            # Save the updated item
            item.save()

            return JsonResponse({'message': 'Item restocked successfully!'}, status=200)
        except Item.DoesNotExist:
            return JsonResponse({'error': 'Item not found.'}, status=404)
    else:
        return JsonResponse({'error': 'Invalid quantity.'}, status=400)


def upload_items(request):
    if request.method == "POST":
        excel_file = request.FILES['file']

        # Read Excel File using pandas
        df = pd.read_excel(excel_file)

        for index, row in df.iterrows():
            # Get or create the Type object
            type_obj, created = Type.objects.get_or_create(name=row['Type Name'])

            # Create Item object without saving it yet
            item = Item(
                name=row['Item Name'],
                type=type_obj,
                price=row['Price'],
                stock=row['Stock'],
                features=row['Features'],
                description=row['Description']
            )

            # Generate unique number and QR code
            if not item.unique_number:
                item.unique_number = str(uuid.uuid4().int)[:10]

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(item.unique_number)
            qr.make(fit=True)

            # Create QR code image
            img = qr.make_image(fill='black', back_color='white')
            buffer = BytesIO()
            img.save(buffer, 'PNG')
            file_name = f'{item.unique_number}.png'

            # Save the QR code image to qrcode field
            item.qrcode.save(file_name, File(buffer), save=False)

            # Handle the image upload from the path provided in the Excel file
            image_path = row['Image Path']
            if Path(image_path).exists():
                with open(image_path, 'rb') as img_file:
                    item.image.save(Path(image_path).name, File(img_file))

            # Save the item to the database (which includes QR code)
            item.save()

        return redirect('success_page')

    return render(request, 'myapp/upload_items.html')


def upload_types(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Read the uploaded Excel file
            file = request.FILES['file']
            df = pd.read_excel(file)

            # Loop through the DataFrame and create Type instances
            for index, row in df.iterrows():
                type_name = row['name']
                image_path = row['image_path']

                # Create a new Type object
                new_type = Type(name=type_name)

                # Save the image to the model
                if image_path:
                    new_type.image.save(image_path.split('/')[-1], File(open(image_path, 'rb')), save=False)

                new_type.save()
            return redirect('success_url')  # Replace 'success_url' with your desired redirect URL
    else:
        form = UploadFileForm()
    return render(request, 'myapp/upload_types.html', {'form': form})


def webecom(request):
    items = Item.objects.all()
    cart = request.session.get('cart', [])
    print(f"cart: {cart}")
    return render(request, 'myapp/webecom.html', {'items': items, 'cart': cart})


def add_to_cart(request, item_id):
    cart = request.session.get('cart', [])
    item = Item.objects.get(id=item_id)

    for cart_item in cart:
        if cart_item['id'] == item_id:
            cart_item['quantity'] += 1
            cart_item['price'] += float(item.price)
            break
    else:
        cart.append({
            'id': item.id,
            'name': item.name,
            'price': float(item.price),
            'url': str(item.image.url),
            'unique_number': str(item.unique_number),
            'quantity': 1
        })

    request.session['cart'] = cart
    return JsonResponse({'cart': cart})


def remove_from_cart(request, item_id):
    cart = request.session.get('cart', [])
    for cart_item in cart:
        if cart_item['id'] == item_id:
            cart_item['quantity'] -= 1
            cart_item['price'] -= float(Item.objects.get(id=item_id).price)
            if cart_item['quantity'] == 0:
                cart.remove(cart_item)
            break

    request.session['cart'] = cart
    return JsonResponse({'cart': cart})


def search_items(request):
    query = request.GET.get('query')
    items = Item.objects.filter(name__icontains=query) | Item.objects.filter(type__name__icontains=query)
    items_list = [{'id': item.id, 'name': item.name, 'price': item.price, 'image': item.image.url} for item in items]
    return JsonResponse({'items': items_list})


def camera_view(request):
    worker_name = request.session.get('worker_name', 'Guest')
    return render(request, 'myapp/camera.html', {'worker_name': worker_name})


@csrf_exempt
def save_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Ensure the directory exists
        if not os.path.exists(IMAGE_SAVE_PATH):
            os.makedirs(IMAGE_SAVE_PATH)

        # Generate a unique file name for the image
        file_name = "captured_image.jpg"
        file_path = os.path.join(IMAGE_SAVE_PATH, file_name)

        # Save the image to the local folder
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # Perform face detection and recognition
        return detect_and_recognize_face(file_path)

    return JsonResponse({'status': 'Invalid request'}, status=400)


def detect_faces_and_get_embedding(image_path):
    try:
        # Load Haar Cascade for face detection
        haar_cascade_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'myapp',
                                         'haarcascade_frontalface_default.xml')
        haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

        if haar_cascade.empty():
            raise ValueError("Error loading Haar cascade.")

        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Error loading image.")

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

        if len(faces) > 0:
            # Assuming we take the first detected face
            x, y, w, h = faces[0]
            face = img[y:y + h, x:x + w]

            # Convert the detected face to PIL format
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Load the embeddings instance
            ibed = imgbeddings()

            # Calculate embeddings for the detected face
            embedding = ibed.to_embeddings(face_pil)

            return (embedding, (x, y, w, h))
        else:
            return (None, None)  # No faces detected

    except Exception as e:
        print(f"Error during face detection/embedding calculation: {e}")
        return (None, None)  # Handle errors gracefully


def detect_and_recognize_face(image_path):
    try:
        embedding, face_coordinates = detect_faces_and_get_embedding(image_path)
        embedding = embedding.tobytes()
        embedding = np.frombuffer(embedding, dtype=np.float32)
        # print(f"Face embedding: {embedding[:20]}")
        if embedding is None:
            return JsonResponse({'status': 'No face detected'})

        closest_worker = None
        min_distance = float('inf')

        # Compare embeddings with stored worker embeddings
        for wrk in worker.objects.all():
            if wrk.face_encodings:

                stored_embedding = np.frombuffer(wrk.face_encodings, dtype=np.float32)
                # print(f"wrk embedding of {wrk.name}: {stored_embedding[:20]}")
                distance = np.linalg.norm(embedding - stored_embedding)
                print(distance)

                if distance < min_distance:
                    min_distance = distance
                    closest_worker = wrk

        # Set a threshold for determining if a match is found
        threshold = 10
        if closest_worker and min_distance < threshold:
            recognition_status = f"{closest_worker.name}"
            print(recognition_status)
            return JsonResponse({'status': recognition_status})
        else:
            recognition_status = None

        # Draw a rectangle around the face if coordinates are available
        if face_coordinates:
            x, y, w, h = face_coordinates
            img = cv2.imread(image_path)  # Reload image for drawing rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return JsonResponse({'status': recognition_status})

    except Exception as e:
        print(f"Error during face detection/recognition: {e}")
        return JsonResponse({'status': None})


def register_worker(request):
    if request.method == 'POST':
        form = WorkerRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            wrk = form.save(commit=False)

            # Process the uploaded image
            image_file = request.FILES['image']
            file_path = f'media/{image_file.name}'  # Save the uploaded image

            with open(file_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Load the image and detect face
            embedding, _ = detect_faces_and_get_embedding(file_path)

            if embedding is not None:
                # Convert numpy array to binary format for storage
                wrk.face_encodings = embedding.tobytes()  # Store as binary
                wrk.save()

                return redirect('worker_login')  # Redirect to login page after registration
    else:
        form = WorkerRegistrationForm()
    return render(request, 'myapp/register.html', {'form': form})


def home(request):
    request.session['cart'] = []
    return render(request, 'myapp/home.html')


class AllOrdersView(ListView):
    model = Order
    template_name = 'myapp/all_orders.html'  # Specify your template
    context_object_name = 'orders'

    def get_queryset(self):
        return Order.objects.select_related('item').prefetch_related('customer').all()


def close_ticket(request, ticket_id):
    print(ticket_id)
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    if request.method == 'POST':
        ticket.done = True
        ticket.save()
        return redirect('ticket_details', ticket_id=ticket.ticket_id)


def ticket_details(request, ticket_id):
    wrk = request.GET.get('wrk')
    print(wrk)
    # Fetch the ticket details
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    print(ticket_id)
    if str(wrk).strip() != str(ticket_id).strip():
        wrk = None
        print(wrk)

    # Fetch the related order
    order = ticket.order
    print(order)

    # Fetch the review for the order
    review = Review.objects.filter(order=order).first()  # Assuming there's at most one review per order

    # Fetch the receipt that includes this order
    receipt = Receipt.objects.filter(
        orders=order).first()  # Assuming a Many-to-Many relationship for receipts and orders

    # Fetch the worker assigned to this ticket, if any
    assigned_worker = ticket.associates.first() if ticket.assigned else None

    context = {
        'ticket': ticket,
        'order': order,
        'review': review,
        'receipt': receipt,
        'assigned_worker': assigned_worker,
        'wrk': wrk if wrk else None
    }

    return render(request, 'myapp/ticket_details.html', context)


def all_tickets_view(request):
    tickets = Ticket.objects.all().select_related('order', 'item')  # Fetch all tickets with related order and item
    reviews = Review.objects.all()  # Fetch all reviews (optional: filter by ticket or order)

    context = {
        'tickets': tickets,
        'reviews': reviews
    }
    return render(request, 'myapp/all_tickets.html', context)


@csrf_exempt  # Disable CSRF protection for this view (handle CSRF tokens manually in production)
def unassign_ticket(request, ticket_id):
    if request.method == 'POST':
        try:
            # Retrieve the ticket
            ticket = Ticket.objects.get(ticket_id=ticket_id)

            # If the ticket is not assigned, return an error
            if not ticket.assigned:
                return JsonResponse({'success': False, 'message': 'Ticket is not assigned.'})

            # Fetch all workers assigned to this ticket and unassign the ticket
            workers_assigned = worker.objects.filter(tickets=ticket)
            if workers_assigned.exists():
                for assigned_worker in workers_assigned:
                    assigned_worker.tickets.remove(ticket)  # Remove the ticket from each assigned worker

            # Mark the ticket as unassigned
            ticket.assigned = False
            ticket.save()

            return JsonResponse({'success': True, 'message': 'Ticket unassigned successfully.'})

        except Ticket.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Ticket not found.'})

    # If not a POST request, return an error
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})


@csrf_exempt  # Disable CSRF protection for this view (since we are using JavaScript, but handle CSRF manually for security)
def assign_ticket(request, ticket_id):
    if request.method == 'POST':
        try:
            # Retrieve the ticket
            ticket = Ticket.objects.get(ticket_id=ticket_id)

            # If the ticket is already assigned, return an error
            if ticket.assigned:
                return JsonResponse({'success': False, 'message': 'Ticket is already assigned.'})

            # Mark the ticket as assigned
            ticket.assigned = True
            ticket.save()

            # Fetch all workers and assign the ticket to a random worker
            all_workers = worker.objects.all()
            if all_workers.exists():
                assigned_worker = random.choice(all_workers)
                assigned_worker.tickets.add(ticket)  # Assign the ticket to the worker

                return JsonResponse({'success': True, 'worker': assigned_worker.name})
            else:
                return JsonResponse({'success': False, 'message': 'No workers found.'})

        except Ticket.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Ticket not found.'})

    # If not a POST request, return an error
    return JsonResponse({'success': False, 'message': 'Invalid request method.'})


def item_info(request, item_id, identifier):
    item = Item.objects.get(id=item_id)
    type = item.type
    reviews = Review.objects.filter(order__item=item).select_related('order')
    tickets = Ticket.objects.filter(item=item) if identifier == 1 else 2

    context = {
        'item': item,
        'reviews': reviews,
        'tickets': tickets,
        'typenm': type.name,
        'typeid': type.id,
    }
    return render(request, 'myapp/item_info.html', context)


def get_orders(request, unique_number):
    print(unique_number)
    # Fetch all orders that have the specific item based on the unique number
    orders = Order.objects.filter(item__unique_number=unique_number)  # Adjust based on your model
    stock_left = get_object_or_404(Item, unique_number=unique_number).stock  # Get stock for the item

    order_data = []
    total_quantity = 0
    graph_data = defaultdict(int)  # To store quantities grouped by month

    for order in orders:
        order_data.append({
            'created_at': order.created_at,
            'quantity': order.quantity
        })
        total_quantity += order.quantity

        # Group by month and year
        month_key = order.created_at.strftime('%Y-%m')
        graph_data[month_key] += order.quantity

    # Prepare graph data for JSON response
    graph_data_json = {key: value for key, value in graph_data.items()}
    print(graph_data_json)
    print(stock_left)
    return JsonResponse({
        'orders': order_data,
        'stock': stock_left,
        'graph_data': graph_data_json,
        'total_quantity': total_quantity
    })


@csrf_exempt
def delete_review(request):
    if request.method == 'POST':
        try:
            # Parse the incoming JSON request data
            data = json.loads(request.body)
            order_id = data.get('order_id', None)

            if order_id:
                try:
                    # Fetch the order associated with the order_id
                    order = Order.objects.get(order_id=order_id)
                    print(order_id, order)
                    # Find the review associated with this order
                    try:

                        review = Review.objects.get(order=order)
                        try:
                            ticket = Ticket.objects.get(order=order)
                        except Ticket.DoesNotExist:
                            ticket = None
                        # Delete the review
                        review.delete()
                        # Delete the Ticket
                        if ticket:
                            ticket.delete()

                        # Uncheck the reviewed field in the order and save
                        order.reviewed = False
                        order.save()

                        return JsonResponse({"success": True, "message": "Review deleted successfully."})
                    except Review.DoesNotExist:
                        return JsonResponse({"success": False, "message": "Review not found."}, status=404)
                except Order.DoesNotExist:
                    return JsonResponse({"success": False, "message": "Order not found."}, status=404)
            else:
                return JsonResponse({"success": False, "message": "Order ID not provided."}, status=400)

        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({"success": False, "message": "Error processing delete request."}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request."}, status=405)


def extract_json_from_response(response):
    try:
        # Regex pattern to find JSON-like content (allow some non-JSON text before/after)
        json_match = re.search(r'{.*}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None


# Function to speak out loud using gTTS
@csrf_exempt
def give_review(request):
    if request.method == 'POST':
        ticket = None
        try:
            data = json.loads(request.body)
            review_text = data.get('review', '')
            print("Received review:", review_text)

            order_id = data.get('order_id', None)  # Ensure the order_id is passed in the request
            cn = data.get('eml', None)
            print("Received order_id:", order_id)

            # Fetch the order associated with the review
            if order_id:
                print("order id is there")
                try:
                    print("trying to find the order with order id")
                    order = Order.objects.get(order_id=order_id)
                    item = order.item
                    total = item.price * order.quantity

                    if item.description:
                        relation_test = f"'question': \"Is the review: \"{review_text}\" even slightly related to the item or item parts or item description in the context: \"{item.description}\", Reply Yes or No "
                        relation = generate_response(relation_test)
                        print(relation)
                        if str(relation).lower() == 'no':
                            return JsonResponse({"success": False, "message": "Please give a relevant review."})
                    language_test = f"'question': \"Does the review: \"{review_text}\" contain any offensive language, such as profanity, disrespectful words, or anything that shows bias against race, religion, gender, or sexual orientation, even in the slightest? Please respond with 'yes' or 'no'.\""
                    language = generate_response(language_test)
                    print("language")
                    print(language)
                    if str(language).lower() == 'yes':
                        return JsonResponse({"success": False,
                                             "message": "We value all feedback, but please ensure that your review is respectful and free of offensive language. Kindly revise your review to better reflect our community standards."})
                    senti = str(return_sentiment(review_text))
                    print(f"The sentiment of the Review {review_text} is: {senti}")

                    if senti == 'negative':
                        # Create a new Ticket linked to the order and item
                        ticket = Ticket.objects.create(order=order, item=item)
                        ticket_id = ticket.ticket_id
                        additional_message = f"Add to the message that we have created a ticket with ticket id: {ticket_id} and someone will be in touch with you soon."
                    else:
                        additional_message = ""

                    print(order)
                except Order.DoesNotExist:
                    return JsonResponse({"success": False, "message": "Order not found."}, status=404)
            else:
                return JsonResponse({"success": False, "message": "Order ID not provided."}, status=400)

            prompt = f"""
                        Features:
                        Performance - total: 100
                        Design - total: 100
                        Price - total: 100
                        Quality - total: 100
                        Ease of use - total: 100
                        Customer support - total: 100

                        Review: {review_text}
                        Based on the review and product features provided above, please do the following:

                        1. Impact Assessment: 
                            - Each Feature: Assign a percentage impact to each feature based on the review. If the review does not affect a particular feature, please leave it unchanged. Consider how the features might impact each other differently based on the review text. 
                            - "Reason": Provide a single reason text to nalyse how it impacts the features. 
                            - "Stars": Give a stars rating out of 5(maximum) 1(minimum), be mild when assigning star rating for the product.

                        2. Draft a Reply: Write a well-structured response addressing the review, incorporating the impact percentages for each feature.

                        {additional_message}  # Insert the additional message if sentiment is negative

                        Return with just a JSON:
                        {{
                            "Impact assessment": {{
                                "Features": "Percentages",
                                "Reason": "Reasons for the impact",
                                "Stars": "Give stars rating based on the review and be mild"
                            }},
                            "Draft Reply": "Your well-structured response here"
                        }}
                        """

            print(prompt)
            # Generate response using the LLM
            response = generate_response(prompt)
            print(str(response))
            # Example response processing (assuming the LLM gives a JSON formatted string)
            json_response = extract_json_from_response(str(response))
            draft_reply = json_response.get('Draft Reply', '')
            Impact_assessment = json_response.get('Impact assessment', '')
            print(Impact_assessment)
            print(Impact_assessment['Stars'])
            st = int(Impact_assessment['Stars'])
            review_comments = str(Impact_assessment['Features']) + str(Impact_assessment['Reason'])
            print(f"review_comments: {review_comments}")
            # Create the Review object with the order, review_text, and draft_reply
            review = Review.objects.create(
                order=order,
                review_text=review_text,
                reply_text=draft_reply,  # Save the generated reply as reply_text
                ticket=ticket if ticket else None,
                stars=st,
                review_comments=review_comments

            )
            if not order.customer:
                customer, created = Customer.objects.get_or_create(email=cn)
                customer.total_spent = Decimal(str(customer.total_spent)) + Decimal(total)
                order.customer = customer
                order.email_id = cn
                customer.save()
            # Set the order's reviewed field to True and save the order
            order.reviewed = True

            order.save()

            return JsonResponse({
                "success": True,
                "impact_assessment": json_response.get('Impact assessment', {}),
                "draft_reply": draft_reply
            })

        except Exception as e:
            print(f"Error: {e}")
            if ticket:
                ticket.delete()
            return JsonResponse({"success": False, "message": "Error processing review."}, status=400)

    return JsonResponse({"success": False, "message": "Invalid request."}, status=405)


def process_order_number(request):
    if request.method == 'GET':
        print("hwoihkwh")
        user_input = request.GET.get('input', '')
        user_input = str(user_input).replace(" ", "")
        print(user_input)
        # Check if the input parameter is provided
        if not user_input:
            return HttpResponseBadRequest("Missing input parameter")  # Or redirect as needed
        # print(user_input)
        # # Continue processing with the valid input
        # prompt = f"Given the user input {user_input}, extract only the numbers with no spaces, in same line and return them without any additional text or code."
        # response = generate_response(prompt)
        # print(f"response from hugchat: {response}")
        # # Extract the last four digits from the response
        # order_number = int(str(response)[-4:].strip())  # Adjust as necessary for your actual extraction logic
        # print(f"order number from ai: {order_number}")

        print(f"User input: {user_input}")

        # Extract numbers from the input
        order_number = re.findall(r'\d+', user_input)
        # Check if there's an order with that last 4 digits
        try:
            print(order_number)
            order = Order.objects.get(order_id=str(order_number[0]))
            print(f"Matched order: {order}")

            # Get item details
            item = order.item
            print(f"Item name: {item.name}")

            # Prepare data for the redirect
            data = {
                'order_id': order.order_id,
            }

            print(data)

            # Render the item review order template

            return redirect('item_review_order', order_id=order.order_id)

        except Order.DoesNotExist:
            return HttpResponseBadRequest("Order not found")
        except Exception as e:
            return HttpResponseServerError(f"An unexpected error occurred: {str(e)}")


def generate_date_range(start_date, end_date):
    """
    Generates a list of dates between the start and end dates (inclusive).
    """
    current_date = start_date
    dates = []

    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return dates


def item_review_order(request, order_id=None):
    # Fetch the order using the provided order_id
    order = get_object_or_404(Order, order_id=order_id)
    orders_customer = None
    dates = []
    quantities = []

    if order.email_id:
        orders_customer = Order.objects.filter(customer=order.customer, item=order.item)

        # Extract the earliest and latest order dates
        if orders_customer.exists():
            first_order_date = orders_customer.order_by('created_at').first().created_at
            last_order_date = orders_customer.order_by('created_at').last().created_at

            # Generate all dates between the first and last order dates
            all_dates = generate_date_range(first_order_date, last_order_date)

            # Create a dictionary of existing orders for quick lookup
            order_dict = {order.created_at.strftime('%Y-%m-%d'): order.quantity for order in orders_customer}

            # Populate dates and quantities, filling missing dates with 0
            for date in all_dates:
                dates.append(date)
                quantities.append(order_dict.get(date, 0))  # Get quantity if it exists, otherwise 0

    xd = request.GET.get('xd')
    if xd:
        print("xdddd")

    print(order)  # Change this based on your actual order field
    ticket = None
    try:
        # Try to get the ticket
        print("afjbajk")
        print(order_id)
        ticket = Ticket.objects.get(order=order)
        print(f"ticket: {ticket}")
    except Ticket.DoesNotExist:
        # If no ticket exists, set ticket to None or handle as needed
        ticket = None

    # Default context values for orders
    context = {
        'item_name': order.item.name,  # Assuming your Order model has an item with a name
        'item_unique_number': order.item.unique_number,
        'item_image': order.item.image.url,  # Assuming item.image is an ImageField
        'item_features': order.item.features,
        'order_id': order.order_id,
        'order_quantity': order.quantity,
        'order_reviewed': order.reviewed,
        'order_email': order.email_id if order.email_id else None,
        'ticket_status': ticket.assigned if ticket else None,
        'ticket_id': ticket.ticket_id if ticket else None,
        'ticket_done': ticket.done if ticket else None,
        'xd': xd if xd else None,
        'orders_customer': orders_customer if orders_customer else None,
        'dates': dates if dates else None,  # Updated dates including missing ones
        'quantities': quantities if quantities else None  # Updated quantities with zeros for missing dates
    }

    # If the order has been reviewed, fetch the review and its reply
    if order.reviewed:
        try:
            review = Review.objects.get(order=order)  # Assuming the Review model has a ForeignKey to the order
            context['review_text'] = review.review_text  # Assuming 'text' is the field name for the review content
            context['reply_text'] = review.reply_text  # Assuming 'reply' is the field name for the reply content
        except Review.DoesNotExist:
            # If somehow the review is not found, ensure it does not break the page
            context['review_text'] = None
            context['reply_text'] = None

    return render(request, 'myapp/item_review_order.html', context)


def scan_qrcode(request):
    cart = request.session.get('cart', [])

    if request.method == 'POST':

        # Check if the request is to remove an item from the cart
        remove_item_index = request.POST.get('remove_item_index')
        if remove_item_index is not None:
            index_to_remove = int(remove_item_index)
            if 0 <= index_to_remove < len(cart):
                cart.pop(index_to_remove)
                request.session['cart'] = cart  # Update the session cart
                # No need to redirect since we're removing an item
        else:
            # Handle adding item logic
            unique_number = request.POST.get('unique_number1')

            if unique_number:
                item = get_object_or_404(Item, unique_number=unique_number)
                print(unique_number)
                print(item.image.url)

                if not item.available:
                    # If item is out of stock, return an error message to the template
                    error_message = f"The item '{item.name}' is out of stock."
                    total = sum(item['price'] for item in cart)
                    Items = Item.objects.all()
                    return render(request, 'myapp/scan_portal.html', {
                        'cart': cart,
                        'total': total,
                        'Items': Items,
                        'error_message': error_message
                    })

                # Check if the item is already in the cart
                item_in_cart = next((entry for entry in cart if entry['unique_number'] == unique_number), None)

                if item_in_cart:
                    # Item exists in the cart, increase quantity and update price
                    item_in_cart['quantity'] += 1
                    future_quanity = item.stock - item_in_cart['quantity']
                    if future_quanity < 0:
                        error_message = f"The item '{item.name}' will be out of stock, if you add one more {item.name}."
                        item_in_cart['quantity'] -= 1
                        total = sum(item['price'] for item in cart)
                        Items = Item.objects.all()
                        return render(request, 'myapp/scan_portal.html', {
                            'cart': cart,
                            'total': total,
                            'Items': Items,
                            'error_message': error_message
                        })
                    item_in_cart['price'] = float(item.price) * item_in_cart['quantity']
                else:
                    # Item not in the cart, add it
                    cart.append({
                        'name': item.name,
                        'price': float(item.price),
                        'url': str(item.image.url),
                        'unique_number': str(item.unique_number),
                        'quantity': 1
                    })

                # Update the session cart
                request.session['cart'] = cart
                return redirect('scan_portal')

    # Recalculate the total after any changes to the cart
    total = sum(item['price'] for item in cart)
    Items = Item.objects.all()
    return render(request, 'myapp/scan_portal.html', {'cart': cart, 'total': total, 'Items': Items})


def checkout(request):
    cart = request.session.get('cart', [])
    if len(cart) == 0:
        return redirect('webecom')
    total = sum(item['price'] for item in cart)
    tax = total * 0.10
    total_with_tax = total + tax
    request.session['cart'] = []
    print(cart)
    return render(request, 'myapp/checkout.html', {'cart': cart, 'total': total_with_tax, 'tax': tax})


def generate_unique_code():
    # Generate a random 4-digit number
    code = random.randint(1000, 9999)

    # Check if the code already exists in the Order model
    while Order.objects.filter(code=code).exists():
        code = random.randint(1000, 9999)  # Regenerate if not unique

    return code


def confirm_payment(request):
    if request.method == 'POST':
        # Retrieve cart data
        cart_items = request.POST.getlist('cart_items')
        cart_quantities = request.POST.getlist('cart_quantities')

        # Create a new Receipt
        receipt = Receipt.objects.create()
        orders_data = []  # List to hold order details for the bill
        email = request.POST.get('email', None)
        tax = request.POST.get('tax')
        total = request.POST.get('total')
        customer = None
        if email:
            customer, created = Customer.objects.get_or_create(email=email)
            # Update the customer's total spent if not newly created
            customer.total_spent = Decimal(str(customer.total_spent)) + Decimal(total)
            customer.save()
        for unique_number, quantity in zip(cart_items, cart_quantities):
            if not unique_number:  # Check if unique_number is empty
                continue  # Skip this item if the unique_number is not valid

            try:
                # Use unique_number instead of id to get the Item
                item = Item.objects.get(unique_number=unique_number)

                # Generate unique 12-digit order ID
                order_id = ''.join([str(random.randint(0, 9)) for _ in range(12)])

                # Create the Order
                order = Order.objects.create(
                    order_id=order_id,
                    item=item,
                    quantity=int(quantity),
                    email_id=email if email else None,
                    customer=customer
                )
                order.save()
                # Add the order to the receipt
                receipt.orders.add(order)

                # Update the stock
                item.stock -= int(quantity)
                item.save()

                # Append order data for display
                orders_data.append({
                    'order_id': order_id,
                    'item_name': item.name,
                    'quantity': quantity,
                    'price': item.price,
                    'total': int(quantity) * item.price
                })

            except Item.DoesNotExist:
                print(f"Item with unique_number {unique_number} does not exist.")  # Log the error

        receipt.total_tax = tax
        receipt.total_bill = total
        # Save the receipt after adding all orders
        receipt.save()
        # Clear the cart session
        request.session['cart'] = []

        # Pass the orders data to the template for display
        return render(request, 'myapp/payment_success.html',
                      {'orders_data': orders_data, 'receipt_id': receipt.id, 'time': receipt.created_at,
                       'email': email if email else None, 'tax': tax, 'total': total})

    return redirect('checkout')


def reviewspage(request):
    # Get all the orders linked to the user (this example assumes the user has an ID, adjust as needed)
    orders = Order.objects.all()  # Adjust this query for your user logic
    return render(request, 'myapp/reviewspage.html', {'orders': orders})


def admin_portal(request):
    if request.method == 'POST':
        # Handle deletion
        if 'type_id' in request.POST:
            type_id = int(request.POST['type_id'])
            try:
                type_instance = Type.objects.get(id=type_id)
                Item.objects.filter(type=type_instance).delete()  # Delete related items
                type_instance.delete()  # Delete the type
                return redirect('admin_portal')  # Redirect to the same page
            except Type.DoesNotExist:
                return JsonResponse({'error': 'Type not found'}, status=404)

    # Handle GET request
    types = Type.objects.all()
    types_with_items = []
    for type_instance in types:
        items = Item.objects.filter(type=type_instance).values_list('name', flat=True)
        types_with_items.append({
            'type': type_instance,
            'items': list(items)
        })
    print(types_with_items)
    return render(request, 'myapp/admin_portal.html', {'types_with_items': types_with_items})


def add_type(request):
    if request.method == 'POST':
        form = TypeForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('admin_portal')
    else:
        form = TypeForm()
    return render(request, 'myapp/add_type.html', {'form': form})


def type_items(request, type_id):
    if request.method == 'POST':
        # Handle deletion
        if 'item_id' in request.POST:
            item_id = int(request.POST['item_id'])
            item = get_object_or_404(Item, id=item_id)
            item.delete()  # Delete the item
            return redirect('type_items', type_id=type_id)  # Redirect to the same page

    items = Item.objects.filter(type_id=type_id)
    type_instance = get_object_or_404(Type, id=type_id)
    return render(request, 'myapp/type_items.html', {'type': type_instance, 'items': items})


def add_item(request, type_id):
    type_instance = get_object_or_404(Type, id=type_id)

    if request.method == 'POST':
        print(request.POST)  # Debugging line to see the submitted data
        form = ItemForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)
            item.type = type_instance

            # Get the features from the request as a list
            features = request.POST.getlist('features')  # Retrieve all features
            print(f"features list: {features}")  # Debugging line

            # Filter out empty values and join into a single string
            features_string = ','.join(filter(None, features)).strip()  # Only join non-empty values
            print(f"features string: {features_string}")  # Debugging line

            # Save the features to the item
            item.features = features_string

            item.save()
            return redirect('type_items', type_id=type_instance.id)
        else:
            print(form.errors)  # Print any form validation errors
    else:
        form = ItemForm()

    return render(request, 'myapp/add_item.html', {'form': form, 'type': type_instance})


def worker_login(request):
    form = WorkerLoginForm(request.POST or None)
    error_message = None

    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            try:
                work = worker.objects.get(username=username)
                if work.password == password:
                    request.session['worker_id'] = work.worker_id  # Store worker ID in session
                    request.session['worker_name'] = work.name
                    return redirect('camera_view')  # Redirect to the ticket display page
                else:
                    error_message = "Invalid password"
            except worker.DoesNotExist:
                error_message = "Worker not found"

    return render(request, 'myapp/worker_login.html', {'form': form, 'error_message': error_message})


def worker_tickets(request):
    worker_id = request.session.get('worker_id')
    if not worker_id:
        return redirect('worker_login')  # Redirect to login if no worker is logged in

    wrk = worker.objects.get(worker_id=worker_id)
    tickets = wrk.tickets.all()

    return render(request, 'myapp/worker_tickets.html', {'worker': wrk, 'tickets': tickets})


ENCODINGS_FILE = 'face_encodings.pkl'  # File to save encodings


def worker_profile(request):
    # Fetch the worker's details using the session worker_id
    worker_id = request.session.get('worker_id')
    if worker_id:
        work = worker.objects.get(worker_id=worker_id)
        context = {
            'worker': work
        }
        return render(request, 'myapp/profile_worker.html', context)
    else:
        return redirect('worker_login')  # Redirect to login if session does not exist


# Logout View
def worker_logout(request):
    logout(request)  # Clear the session
    return redirect('worker_login')


def worker_data(request):
    workers = worker.objects.all()
    return render(request, 'myapp/worker_data.html', {'workers': workers})


def customer_data(request):
    customers = Customer.objects.all()
    return render(request, 'myapp/customer_data.html', {'customers': customers})


def get_customer_orders(request, customer_id):
    try:
        customer = Customer.objects.get(id=customer_id)
        orders = Order.objects.filter(customer=customer).select_related('item')

        # Prepare order data for the table
        order_data = []
        for order in orders:
            order_data.append({
                'order_id': order.order_id,
                'item_name': order.item.name,
                'peri': order.item.price,
                'quantity': order.quantity,
                'total_price': order.item.price * order.quantity
            })

        # Get all unique receipts for the customer's orders
        receipts = Receipt.objects.filter(orders__in=orders).distinct()
        receipt_data = []
        for receipt in receipts:
            receipt_data.append({
                'receipt_date': receipt.created_at.strftime('%Y-%m-%d'),
                'total_bill': receipt.total_bill
            })

        return JsonResponse({'orders': order_data, 'receipts': receipt_data}, status=200)

    except Customer.DoesNotExist:
        return JsonResponse({'error': 'Customer not found'}, status=404)


def allitems(request):
    items = Item.objects.all()

    return render(request, 'myapp/allitems.html', {'items': items})
