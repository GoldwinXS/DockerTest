from django.contrib import admin
from .models import Test


# Register your models here.

class TutorialAdmin(admin.ModelAdmin):
    # fields = ['test_title',
    #           'test_content',
    #           'test_published']

    fieldsets = [
        ("title/date", {'fields': ['test_title', 'test_published']}),
        ("content", {"fields": ['test_content']})
    ]


admin.site.register(Test, TutorialAdmin)
