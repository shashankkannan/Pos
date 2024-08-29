from django import forms
from .models import Item, Type


class TypeForm(forms.ModelForm):
    class Meta:
        model = Type
        fields = ['name']


class ItemForm(forms.ModelForm):
    class Meta:
        model = Item
        fields = ['name', 'price', 'stock', 'image']  # Include other fields if necessary
