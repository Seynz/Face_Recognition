# file utama untuk menjalankan aplikasi deteksi wajah
from function import Menu

menu = Menu(use_gpu=False)

while True:
    menu.tampilkan_menu()
    pilihan = input("Pilih menu (0-2): ")

    if pilihan == "1":
        menu.deteksi_wajah()
    elif pilihan == "2":
        menu.tambah_wajah()
    elif pilihan == "0":
        print("Terima kasih, program selesai.")
        break
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")