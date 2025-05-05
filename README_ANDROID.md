# بناء تطبيق سكارارا للأندرويد

هذا الملف يشرح كيفية بناء وتثبيت تطبيق سكارارا على نظام الأندرويد.

## المتطلبات

1. نظام Linux أو WSL على Windows
2. Python 3.7 أو أحدث
3. Buildozer
4. Android SDK و NDK
5. Java JDK 8

## خطوات البناء

### 1. تثبيت المتطلبات على نظام Ubuntu/Debian

```bash
# تثبيت المتطلبات الأساسية
sudo apt update
sudo apt install -y git zip unzip openjdk-8-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev

# تثبيت المتطلبات الإضافية
sudo apt install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

# تثبيت Buildozer
pip3 install --user --upgrade buildozer

# تثبيت Cython (مطلوب لـ Buildozer)
pip3 install --user --upgrade Cython==0.29.19
```

### 2. بناء التطبيق

```bash
# تهيئة Buildozer (إذا لم تكن قد قمت بذلك من قبل)
buildozer init

# بناء التطبيق
buildozer -v android debug
```

سيقوم Buildozer بتنزيل وتثبيت Android SDK و NDK تلقائيًا إذا لم تكن موجودة. قد تستغرق هذه العملية وقتًا طويلاً في المرة الأولى.

### 3. تثبيت التطبيق على جهاز الأندرويد

قم بتوصيل جهاز الأندرويد بالكمبيوتر وتأكد من تفعيل وضع المطور وتصحيح الأخطاء عبر USB.

```bash
# تثبيت التطبيق على الجهاز المتصل
buildozer -v android deploy run
```

أو يمكنك نسخ ملف APK من المجلد `bin` ونقله إلى جهاز الأندرويد وتثبيته يدويًا.

## استكشاف الأخطاء وإصلاحها

### مشاكل شائعة وحلولها

1. **خطأ في تنزيل Android SDK أو NDK**:
   ```bash
   # حذف مجلد .buildozer وإعادة المحاولة
   rm -rf .buildozer
   buildozer -v android debug
   ```

2. **مشاكل في تثبيت المكتبات**:
   ```bash
   # تحديث pip وإعادة المحاولة
   pip3 install --user --upgrade pip
   buildozer -v android debug
   ```

3. **مشاكل في تشغيل التطبيق**:
   ```bash
   # عرض سجلات التطبيق
   buildozer android logcat
   ```

## الأذونات المطلوبة

التطبيق يحتاج إلى الأذونات التالية:

1. **الكاميرا**: لالتقاط الصور
2. **قراءة التخزين الخارجي**: لتحميل الصور من الجهاز
3. **كتابة التخزين الخارجي**: لحفظ النتائج والصور المعالجة

تأكد من منح هذه الأذونات للتطبيق عند تشغيله لأول مرة.

## ملاحظات إضافية

- تم تعطيل OCR (التعرف الضوئي على الحروف) على نظام الأندرويد، وبدلاً من ذلك يتم استخدام طريقة بديلة للكشف عن الثقوب المرقمة.
- للحصول على أفضل النتائج، استخدم صورًا ذات إضاءة جيدة وتباين واضح بين الثقوب والخلفية.
- يمكنك ضبط معلمات الكشف في علامة التبويب "متقدم" للحصول على نتائج أفضل.
