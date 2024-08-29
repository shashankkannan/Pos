import uuid
import qrcode
from django.db import models
from django.core.files import File
from io import BytesIO


class Type(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class Item(models.Model):
    name = models.CharField(max_length=200)
    type = models.ForeignKey(Type, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.IntegerField(default=100)
    available = models.BooleanField(default=True)
    unique_number = models.CharField(max_length=100, unique=True, blank=True)
    qrcode = models.ImageField(upload_to='qrcodes/', blank=True, null=True)
    image = models.ImageField(upload_to='items/', blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.unique_number:
            # Generate a unique number
            self.unique_number = str(uuid.uuid4().int)[:10]  # You can customize this

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(self.unique_number)
        qr.make(fit=True)

        # Create an image from the QR code instance
        img = qr.make_image(fill='black', back_color='white')

        # Save the image in a BytesIO buffer
        buffer = BytesIO()
        img.save(buffer, 'PNG')
        file_name = f'{self.unique_number}.png'

        # Save the image to the qrcode field
        self.qrcode.save(file_name, File(buffer), save=False)

        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
