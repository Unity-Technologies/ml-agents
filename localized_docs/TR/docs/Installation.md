# Kurulum

ML-Agents Araç Seti birkaç bileşen içermektedir:

- Unity paketi ([`com.unity.ml-agents`](../com.unity.ml-agents/)) Unity sahnenize entegre edilecek Unity C# SDK içerir.
- Python paketleri:
  - [`mlagents`](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/ml-agents) Unity sahnenizdeki davranışları eğitmenizi sağlayan makine öğrenimi algoritmalarını içerir. Bu nedenle `mlagents` paketini kurmanız gerekecek.
  - [`mlagents_envs`](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/ml-agents-envs) Unity sahnesiyle etkileşime girmek için Python API içermektedir. Unity sahnesi ile Python makine öğrenimi algoritmaları arasında veri mesajlaşmasını kolaylaştıran temel bir katmandır.
    Sonuç olarak, `mlagents,` `mlagents_envs` apisine bağımlıdır.
  - [`gym_unity`](https://github.com/Unity-Technologies/ml-agents/tree/release_7_docs/gym-unity) OpenAI Gym arayüzünü destekleyen Unity sahneniz için bir Python kapsayıcı sağlar.
  <!-- düzenle learning-envir... -->
- Unity [Project](../Project/) klasörü
  [örnek ortamlar](Learning-Environment-Examples.md) ile başlamanıza yardımcı olacak araç setinin çeşitli özelliklerini vurgulayan sahneler içermektedir.

ML-Agents Toolkit 'i kurmanız için gerekenler:

- Unity yükleyin (2022.3 veya daha sonraki bir sürüm)
- Python yükleyin (3.10.12 veya daha yüksek bir sürüm)
- Bu depoyu klonlayın (İsteğe bağlı)
  - __Not:__ Depoyu klonlamazsanız, örnek ortamlara ve eğitim yapılandırmalarına erişemezsiniz. Ek olarak, [Başlangıç Rehberi](Getting-Started.md) depoyu klonladığınızı varsayar.
- `com.unity.ml-agents` ML-Agents Unity paketini yükleyin.
- `mlagents` Python paketini yüklemek.

### **Unity 2022.3** veya Sonraki Bir Sürüm Yükleyin

[İndir](https://unity3d.com/get-unity/download) ve Unity'i yükle. Şiddetli bir şekilde Unity Hub üzerinden kurmanızı ve bu şekilde birden fazla Unity sürümünü yönetmenizi öneriyoruz.

### **Python 3.10.12** veya Daha Yüksek Bir Sürüm Yükleyin

Python 3.10.12 veya daha yüksek bir sürümü [yüklemenizi](https://www.python.org/downloads/) öneriyoruz. Eğer, Windows kullanıyorsanız, lütfen x86-64 versiyonunu kurun ve asla sadece x86 isimli versiyonu kurmayın. Python ortamınız `pip3` içermiyorsa, [talimatları](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers) takip ederek yükleyebilirsiniz.

Windows'ta Anaconda kurulumu için destek sağlamıyor olsak da,
önceki [Windows için Anaconda Yüklemesi (Kullanımdan Kaldırılan) Rehberine](Installation-Anaconda-Windows.md) bakabilirsiniz.

### ML-Agent Toolkit Deposunu Klonlayın (İsteğe Bağlı)

Artık Unity ve Python'u kurduğunuza göre, Unity ve Python paketlerini kurabilirsiniz. Bu paketleri yüklemek için depoyu klonlamanıza gerek yoktur, ancak örnek ortamlarımızı ve eğitim yapılandırmalarımızı bunlarla denemek için indirmek isterseniz depoyu klonlamayı seçebilirsiniz (bizim bazı eğitim serilerimiz / rehberlerimiz örnek ortamlarımıza erişiminizin olduğunu varsayar).

```sh
git clone --branch release_8 https://github.com/Unity-Technologies/ml-agents.git
```

`--branch release_8` seçeneği en son kararlı sürümün etiketine geçecektir. Bunu ihmal ederek yüklerseniz, potansiyel olarak kararsız olan `master` dalı seçilecektir.

#### Gelişmiş: Geliştirme için Yerel Kurulum

Eğer, ML-Agent Toolkit'i amaçlarınız için değiştirmeyi veya genişletmeyi planlıyorsanız, depoyu klonlamanız gerekecektir. Bu değişikliklere tekrar katkıda bulunmayı planlıyorsanız, `master` dalını klonladığınızdan emin olun (yukarıdaki komuttan `--branch release_8` 'i çıkarın). ML-Agent Toolkit'e katkıda bulunma hakkında daha fazla bilgi için [Katkı Yönergelerimize](../com.unity.ml-agents/CONTRIBUTING.md) bakın.

### `com.unity.ml-agent` Unity paketini kurun

Unity ML-Agents C# SDK, bir Unity paketidir. Unity `com.unity.ml-agent` paketini [doğrudan Paket Yöneticisi](https://docs.unity3d.com/Manual/upm-ui-install.html) kayıt defterinden kurabilirsiniz.
Lütfen bulmak için 'Advanced' açılır menüsünde 'Preview Packages' seçeneğini etkinleştirdiğinizden emin olun.

**NOT:** Paket Yöneticisi'nde listelenen ML-Agent paketini görmüyorsanız, lütfen [aşağıdaki gelişmiş kurulum talimatlarını]() izleyin.

#### Gelişmiş: Geliştirme için Yerel Kurulum

[Yerel](https://docs.unity3d.com/Manual/upm-ui-local.html) `com.unity.ml-agent` paketini (yeni klonladığınız depodan) projenize şu şekilde ekleyebilirsiniz:

1. Menüye gidin ve `Window` -> `Package Manager` seçeneğini seçin.
1. Paket yöneticisi penceresinde `+` düğmesine tıklayın.
1. `Add package from disk...` 'i seçin.
1. `com.unity.ml-agents` klasörüne gidin.
1. `package.json` dosyasını seçin.

**NOT:** Unity 2018.4'te `+` düğmesi paket listesinin sağ altındadır ve Unity 2019.3'te paket listesinin sol üst tarafındadır.

<p align="center">
  <img src="images/unity_package_manager_window.png"
       alt="Unity Package Manager Window"
       height="300"
       border="10" />
  <img src="images/unity_package_json.png"
     alt="package.json"
     height="300"
     border="10" />
</p>

Dokümantasyonumuzdaki örnekleri takip edecekseniz, Unity'de `Project` klasörünü açabilir ve hemen kurcalamaya başlayabilirsiniz.

### `mlagents` Python paketi kurulumu

`mlagents` Python paketini kurmak, mlagentlerin bağlı olduğu diğer Python paketlerinin kurulmasını içerir. Dolayısıyla, makinenizde bu bağımlılıklardan herhangi birinin daha eski sürümleri zaten kurulu ise kurulum sorunlarıyla karşılaşabilirsiniz. Sonuç olarak, `mlagents` yüklemek için desteklediğimiz yol, Python Sanal Ortamlarından yararlanmaktır. Sanal Ortamlar, her proje için bağımlılıkları izole etmek için bir mekanizma sağlar ve Mac / Windows / Linux'ta desteklenir. [Sanal Ortamlar hakkında özel bir rehber](https://github.com/Unity-Technologies/ml-agents/blob/release_8_docs/docs/Using-Virtual-Environment.md) sunuyoruz.

`mlagents` Python paketini kurmak için sanal ortamınızı etkinleştirin ve komut satırından çalıştırın:

```sh
pip3 install mlagents
```

`mlagents` için klonlanmış depo yerine PyPi'den yüklenmesini şiddetle tavsiye ediyoruz. Eğer doğru bir şekilde kurduysanız, `mlagents-learn --help` komutunu çalıştırabilmeniz gerekir, ardından `mlagents-learn` ile kullanabileceğiniz komut satırı parametrelerini görürsünüz.

`mlagents` paketini yükleyerek, [setup.py dosyasında](https://github.com/Unity-Technologies/ml-agents/blob/release_8_docs/ml-agents/setup.py) listelenen bağımlılıklar da yüklenir. Bunlara
[TensorFlow](https://github.com/Unity-Technologies/ml-agents/blob/release_8_docs/docs/Background-TensorFlow.md) dahildir. (AVX destekli bir CPU gerektirir).

#### Gelişmiş: Geliştirme için Yerel Kurulum

`mlagents` veya `mlagents_envs` üzerinde değişiklikler yapmayı planlıyorsanız, paketleri PyPi yerine klonlanmış depodan yüklemelisiniz. Bunu yapmak için `mlagents` ve `mlagents_env` dosyalarını ayrı ayrı yüklemeniz gerekir. Deponun kök dizininden şunu çalıştırmalısınız:

```sh
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```

Pip'i `-e` bayrağıyla çalıştırmak, Python dosyalarında doğrudan değişiklik yapmanıza ve `mlagents-learn` çalıştırdığınızda bunların yansıtılmasına izin verir. `mlagents` paketi `mlagents_envs` paketine bağlı olduğundan, bu paketleri bu sırayla kurmak önemlidir ve diğer sırayla yüklemek `mlagents_envs` paketini PyPi'den indirecektir.

## Sonraki Adımlar

[Başlangıç Rehberi](Getting-Started.md), Unity içinde ML-Agents Toolkit'i kurma, önceden eğitilmiş bir model çalıştırma, ortamları oluşturma ve eğitimle ilgili birkaç kısa öğretici içermektedir.

## Yardım

ML-Agent'larla ilgili herhangi bir sorunla karşılaşırsanız, [SSS](https://github.com/Unity-Technologies/ml-agents/blob/release_8_docs/docs/FAQ.md) sayfamıza ve [Sınırlamalar](https://github.com/Unity-Technologies/ml-agents/blob/release_8_docs/docs/Limitations.md) sayfalarımıza bakın. Hiçbir şey bulamazsanız, lütfen bir [sorun gönderin(issue)](https://github.com/Unity-Technologies/ml-agents/issues) ve işletim sisteminiz, Python sürümünüz, mümkünse tam hata mesajı ile ilgili bilgileri verdiğinizden emin olun.
