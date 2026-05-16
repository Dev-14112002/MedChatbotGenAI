from streamlit_authenticator.utilities.hasher import Hasher

hashed_password = Hasher.hash("yourpassword")

print(hashed_password)
