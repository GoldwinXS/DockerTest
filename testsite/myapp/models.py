from django.db import models


# Create your models here.

class Test(models.Model):
    """
    Creates a database table. Each variable here will be one col.
    Primary key is automatically assigned

    after this has been setup, one needs to:
    1. make migrations | python manage.py makemigrations
    2. migrate

    """
    test_title = models.CharField(max_length=200)
    test_content = models.TextField()
    test_published = models.DateTimeField('date published')

    def __str__(self):
        return self.test_title


class MLMODEL(models.Model):
    vars = models.IntegerField()

