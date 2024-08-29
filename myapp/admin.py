from django.contrib import admin
from .models import Type, Item


@admin.register(Type)
class TypeAdmin(admin.ModelAdmin):
    list_display = ('name',)


@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'type', 'price', 'stock', 'available', 'unique_number', 'qrcode')
