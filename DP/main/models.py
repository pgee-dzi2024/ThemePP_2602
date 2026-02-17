from django.db import models


class Picture(models.Model):
    name = models.CharField('Име', max_length=100)
    description = models.TextField('Описание')
    picture = models.ImageField('Снимка', upload_to='images/')
    date = models.DateTimeField('Дата на създаване', auto_now_add=True)
    authenticity = models.IntegerField('Достоверност',default=0)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['date']
        verbose_name = 'Снимка'
        verbose_name_plural = 'Снимки'

