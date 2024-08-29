from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from .models import Item, Type
from .forms import ItemForm, TypeForm


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
                cart.append({'name': item.name, 'price': float(item.price), 'url': str(item.image.url)})
                request.session['cart'] = cart
                return redirect('scan_portal')

    # Recalculate the total after any changes to the cart
    total = sum(item['price'] for item in cart)
    return render(request, 'myapp/scan_portal.html', {'cart': cart, 'total': total})


def checkout(request):
    cart = request.session.get('cart', [])
    total = sum(item['price'] for item in cart)
    tax = total * 0.10
    total_with_tax = total + tax
    request.session['cart'] = []
    return render(request, 'myapp/checkout.html', {'cart': cart, 'total': total_with_tax, 'tax': tax})


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
        form = ItemForm(request.POST, request.FILES)
        if form.is_valid():
            item = form.save(commit=False)
            item.type = type_instance
            item.save()
            return redirect('type_items', type_id=type_instance.id)
    else:
        form = ItemForm()
    return render(request, 'myapp/add_item.html', {'form': form, 'type': type_instance})
