from django.urls import path
from . import views

urlpatterns = [
    path('scan-portal/', views.scan_qrcode, name='scan_portal'),
    path('checkout/', views.checkout, name='checkout'),
    path('admin-portal/', views.admin_portal, name='admin_portal'),
    path('admin-portal/add_type/', views.add_type, name='add_type'),
    path('admin-portal/type/<int:type_id>/', views.type_items, name='type_items'),
    path('admin-portal/type/<int:type_id>/add_item/', views.add_item, name='add_item')
]
