# Generated by Django 3.2.25 on 2024-08-28 12:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('stock', models.IntegerField(default=100)),
                ('available', models.BooleanField(default=True)),
                ('qrcode', models.ImageField(blank=True, null=True, upload_to='qrcodes/')),
                ('unique_number', models.CharField(editable=False, max_length=12, unique=True)),
                ('type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myapp.type')),
            ],
        ),
    ]
