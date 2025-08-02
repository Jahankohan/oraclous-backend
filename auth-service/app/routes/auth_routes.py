from fastapi import APIRouter, HTTPException, Request, status, Depends
from app.schema import auth_schemas
from app.services.auth_service import AuthService
from app.core.dependencies import get_user_repository
from fastapi.security import OAuth2PasswordBearer


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
router = APIRouter()

@router.post("/register/", response_model=auth_schemas.Token)
async def register_user(user: auth_schemas.UserCreateWithEmail, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    token_schema = await auth_service.create_user(user=user)
    return token_schema

@router.post("/login/", response_model=auth_schemas.Token)
async def login_user(form_data: auth_schemas.UserLogin, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)
    token_schema = await auth_service.authenticate_user(email=form_data.email, password=form_data.password)
    if not token_schema:
        raise HTTPException(status_code=400, detail="Incorrect email/username or password")
    return token_schema
    

@router.post("/refresh/", response_model=auth_schemas.Token)
async def refresh_token(refresh_token: auth_schemas.RefreshTokenRequest, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    token_schema = await auth_service.verify_refresh_token(refresh_token.refresh_token)
    if not token_schema:
        raise HTTPException(status_code=400, detail="Refresh Token is invalid or expired")
    return token_schema
    

@router.post("/verify-email/")
async def verify_email(email_verification_request: auth_schemas.EmailVerificationRequest, request: Request, token: str = Depends(oauth2_scheme)):
    print("Received token:", token)
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    code = email_verification_request.code
    print("Verification code:", code)
    return await auth_service.verify_user_email(token, code)
    

@router.post("/resend-email-verification/")
async def resend_email_verification(request: Request, token: str = Depends(oauth2_scheme)):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.resend_verification_email(token=token)


@router.post("/forgot-password/")
async def forgot_password(forget_password_req: auth_schemas.ForgotPasswordRequest, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.send_password_reset_email(email=forget_password_req.email)

@router.post("/reset-password/")
async def reset_password(token: str, reset_password_req: auth_schemas.ResetPasswordRequest, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.reset_password(token=token, new_password=reset_password_req.new_password)

@router.post("/change-password/")
async def change_password(change_password_req: auth_schemas.ChangePasswordRequest, request: Request, token: str = Depends(oauth2_scheme)):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    payload = await auth_service.verify_access_token(token)
    email = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    db_user = await auth_service.get_user_by_email(email=email)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return await auth_service.change_password(user=db_user, new_password=change_password_req.new_password)
