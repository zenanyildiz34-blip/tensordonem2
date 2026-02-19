TinyML Tensor System

Bu proje, C dili kullanılarak geliştirilmiş basit bir TinyML Tensor altyapısıdır. Amaç; gömülü sistemlerde ve düşük bellekli ortamlarda kullanılan temel makine öğrenmesi yapılarını (tensor, quantization, float dönüşümleri, matris çarpımı) düşük seviyede göstermektir.Kod ; OpenAI tarafından geliştirilen ChatGPT'den yardım alınarak oluşturulmuştur.Amaç; makine öğrenmesi framework’lerinin arka planda nasıl çalıştığını düşük seviyede göstermek ve özellikle:

-Bellek yönetimi

-Veri tipi optimizasyonu

-Quantization

-Matris çarpımı

-IEEE float temsili
gibi kritik konuları kod üzerinden öğretmektir.

Projenin Mimarisi

Kod şu temel bölümlerden oluşur:

-Veri Tipi Tanımı (enum)

-Bit Seviyesi Erişim (union)

-Tensor Yapısı (struct)

-Float32 ↔ Float16 Dönüşüm

-Bellek Yönetimi

-Quantization

-Matris Çarpımı

-Debug/Test Bölümü (main)

Tensör Nedir?

Tensor, çok boyutlu veri yapısıdır

Quantization Nedir?

Yüksek hassasiyetli float değerleri düşük bitli tiplere dönüştürme işlemidir.
Amaç:

Model boyutunu küçültmek,
RAM kullanımını azaltmak,
CPU hızını artırmak,
Gömülü cihazlarda çalışmak,
TinyML sistemlerde bu zorunludur.


Bu kodun kısaca yaptığı şey, C dili kullanarak basit bir Tensor altyapısı ve TinyML mantığını simüle eden küçük bir matematik motoru oluşturmaktır. Program; farklı veri tiplerinde (float32, float16 ve int8) matris tutabilen bir tensor yapısı tanımlar, bu tensörler için dinamik bellek yönetimi yapar ve temel makine öğrenmesi işlemi olan matris çarpımını gerçekleştirir. Aynı zamanda float32 ile float16 arasında bit seviyesinde dönüşüm yaparak sayının bellekte nasıl temsil edildiğini gösterir ve union kullanımıyla bir float değerin ham (hex) bit karşılığını incelemeye imkân verir.
Kodun önemli kısmı, quantization mantığını uygulamasıdır. Float değerler belirli bir scale ve zero_point kullanılarak int8 formatına dönüştürülür ve gerektiğinde tekrar yaklaşık float değerine çevrilir. Bu, TinyML sistemlerde model boyutunu küçültmek ve bellek kullanımını azaltmak için kullanılan temel tekniktir. Program ayrıca quantize edilmiş tensor ile normal float tensor arasındaki bellek farkını hesaplayarak optimizasyonun etkisini gösterir.

Özetle bu proje; tensör kavramını, veri tipi seçiminin bellek üzerindeki etkisini, quantization’ın nasıl çalıştığını ve matris çarpımının makine öğrenmesindeki rolünü düşük seviyede ve öğretici bir şekilde göstermeyi amaçlayan minimal bir TinyML simülasyonudur.
