FROM python:3.11

# Thiết lập thư mục làm việc
WORKDIR /code

# Sao chép requirements.txt và cài đặt dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Tạo user không có quyền root để chạy ứng dụng (bảo mật)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Thiết lập thư mục làm việc cho user
WORKDIR $HOME/app

# Sao chép source code
COPY --chown=user . $HOME/app

# Expose port cho Chainlit (mặc định là 8000)
EXPOSE 8000

# Command để chạy Chainlit
# Sử dụng host 0.0.0.0 và port 7860 với các flags bổ sung
CMD ["chainlit", "run", "interface/chainlit.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]

# Chạy với 
# docker build -t my-chainlit-app -f interface/Dockerfile .
# docker run -it --rm -p 8000:8000 my-chainlit-app
