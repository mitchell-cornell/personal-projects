import datetime

from django.db import models
from django.utils import timezone
from django.db.models.functions import Concat

class Person(models.Model):
    first_name = models.CharField(max_length = 30)
    last_name = models.CharField(max_length = 30)
    number = models.CharField(max_length = 15)

    full_name = None

    def __str__(self):
        return self.first_name + " " + self.last_name