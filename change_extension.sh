#!/bin/bash

# Kiểm tra xem thư mục có được cung cấp không
if [ -z "$1" ]; then
  echo "Vui lòng cung cấp đường dẫn thư mục."
  exit 1
fi

# Kiểm tra nếu thư mục tồn tại
if [ ! -d "$1" ]; then
  echo "Thư mục không tồn tại."
  exit 1
fi

# Duyệt qua tất cả các tệp trong thư mục
for file in "$1"/*; do
  # Kiểm tra xem tệp có phải là file (không phải thư mục)
  if [ -f "$file" ]; then
    # Lấy tên tệp và thay đổi phần mở rộng thành .jpg
    mv "$file" "${file%.*}.jpg"
  fi
done

echo "Đổi đuôi thành công tất cả các tệp trong thư mục."
