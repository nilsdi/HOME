def prepare_matrikkel_context(matrikkel_version: str, client_purpose: str,
                              client):
    # initializing the objects
    MatrikkelContext = client.factory.create('ns0:MatrikkelContext')
    Timestamp = client.factory.create('ns0:Timestamp')

    # fill these objects with the desired information

    # time fro which we get the matrikkel
    Timestamp.timestamp = matrikkel_version

    # assign the values for MatrikkelContext
    MatrikkelContext.locale = 'no_NO'
    MatrikkelContext.brukOriginaleKoordinater = False

    KoordinatsystemKodeId = client.factory.create('ns14:KoordinatsystemKodeId')
    KoordinatsystemKodeId.value = 10
    MatrikkelContext.koordinatsystemKodeId = KoordinatsystemKodeId
    MatrikkelContext.systemVersion = "trunk"
    MatrikkelContext.klientIdentifikasjon = client_purpose
    MatrikkelContext.snapshotVersion = Timestamp
    MatrikkelContext.systemVersion = "4.3.1.1"
    return Timestamp, MatrikkelContext


def prepare_Kommune(Kommune: str, client):
    # initializing the objects
    KommuneId = client.factory.create('ns21:KommuneId')
    # assign the value for KommuneId
    KommuneId.value = Kommune
    return KommuneId
